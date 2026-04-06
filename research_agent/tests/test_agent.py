from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from research_agent.agent import (
    ResearchAgent,
    _extract_relevance,
    _extract_tags,
    _source_type,
)

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


# --- Unit tests (all external calls mocked) ---


def _make_agent() -> ResearchAgent:
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-fake"}):
        with patch("research_agent.agent.anthropic.Anthropic"):
            return ResearchAgent()


def test_agent_init_raises_without_api_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with patch("research_agent.agent.load_dotenv"):
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                ResearchAgent()


def test_agent_uses_env_model() -> None:
    with patch.dict(
        "os.environ",
        {"ANTHROPIC_API_KEY": "sk-fake", "ANTHROPIC_MODEL": "claude-haiku-4-5-20251001"},
    ):
        with patch("research_agent.agent.anthropic.Anthropic"):
            agent = ResearchAgent()
    assert agent.model == "claude-haiku-4-5-20251001"


def test_agent_defaults_to_sonnet() -> None:
    # Clear ANTHROPIC_MODEL so the default kicks in
    env = {"ANTHROPIC_API_KEY": "sk-fake"}
    with patch.dict("os.environ", env, clear=False):
        import os as _os
        saved = _os.environ.pop("ANTHROPIC_MODEL", None)
        try:
            with patch("research_agent.agent.anthropic.Anthropic"):
                agent = ResearchAgent()
            assert agent.model == "claude-sonnet-4-6"
        finally:
            if saved is not None:
                _os.environ["ANTHROPIC_MODEL"] = saved


def test_process_pdf_single_chunk_one_api_call() -> None:
    """12-page fixture → 1 chunk → 1 Claude call (no merge)."""
    agent = _make_agent()
    fake_note_body = (
        "## Core Claim\nSome claim. #rag #knowledge-graph\n"
        "## Critical Assessment\n**Relevance: 4/5**\n"
    )
    with (
        patch("research_agent.agent.get_page_count", return_value=12),
        patch("research_agent.agent.plan_chunks") as mock_chunks,
        patch("research_agent.agent.extract_pages", return_value="chunk text"),
        patch("research_agent.agent.resolve_topic", return_value="rag"),
        patch("research_agent.agent._vault_topics", return_value=["rag"]),
        patch.object(agent, "_call_claude", return_value=fake_note_body) as mock_call,
    ):
        from research_agent.chunker import Chunk
        mock_chunks.return_value = [Chunk(start_page=1, end_page=12, title="Full Document")]

        note = agent.process_pdf(Path("fake.pdf"))

    mock_call.assert_called_once()  # 1 chunk, no merge call
    assert note.metadata.topic == "rag"
    assert note.metadata.relevance == 4
    assert "rag" in note.metadata.tags
    assert "knowledge-graph" in note.metadata.tags


def test_process_pdf_multi_chunk_calls_merge() -> None:
    """Two chunks → 2 chunk calls + 1 merge call = 3 total."""
    agent = _make_agent()
    fake_body = "## Critical Assessment\n**Relevance: 3/5**\n#rag"

    with (
        patch("research_agent.agent.get_page_count", return_value=25),
        patch("research_agent.agent.plan_chunks") as mock_chunks,
        patch("research_agent.agent.extract_pages", return_value="text"),
        patch("research_agent.agent.resolve_topic", return_value="rag"),
        patch("research_agent.agent._vault_topics", return_value=[]),
        patch.object(agent, "_call_claude", return_value=fake_body) as mock_call,
    ):
        from research_agent.chunker import Chunk
        mock_chunks.return_value = [
            Chunk(start_page=1, end_page=20, title="Part 1"),
            Chunk(start_page=21, end_page=25, title="Part 2"),
        ]
        note = agent.process_pdf(Path("fake.pdf"))

    assert mock_call.call_count == 3  # chunk1, chunk2, merge
    assert note.metadata.relevance == 3


def test_prior_summary_passed_to_second_chunk() -> None:
    """Verify prior_summary from chunk 1 is passed into chunk 2's prompt."""
    agent = _make_agent()
    call_prompts: list[str] = []

    def capture(prompt: str) -> str:
        call_prompts.append(prompt)
        return "## Critical Assessment\n**Relevance: 3/5**\n#ml"

    with (
        patch("research_agent.agent.get_page_count", return_value=25),
        patch("research_agent.agent.plan_chunks") as mock_chunks,
        patch("research_agent.agent.extract_pages", return_value="text"),
        patch("research_agent.agent.resolve_topic", return_value="rag"),
        patch("research_agent.agent._vault_topics", return_value=[]),
        patch.object(agent, "_call_claude", side_effect=capture),
    ):
        from research_agent.chunker import Chunk
        mock_chunks.return_value = [
            Chunk(start_page=1, end_page=20, title="Part 1"),
            Chunk(start_page=21, end_page=25, title="Part 2"),
        ]
        agent.process_pdf(Path("fake.pdf"))

    # call_prompts[0] = chunk 1, call_prompts[1] = chunk 2, call_prompts[2] = merge
    assert "Prior chunks summary" in call_prompts[1]
    assert "Prior chunks summary" not in call_prompts[0]


# --- Helper unit tests ---


def test_extract_relevance_found() -> None:
    body = "Some text\n**Relevance: 4/5**\nmore text"
    assert _extract_relevance(body) == 4


def test_extract_relevance_missing_defaults_to_3() -> None:
    assert _extract_relevance("No relevance line here.") == 3


def test_extract_relevance_clamps() -> None:
    assert _extract_relevance("Relevance: 9/5") == 5
    assert _extract_relevance("Relevance: 0/5") == 1


def test_extract_tags_deduped() -> None:
    body = "#rag #RAG #knowledge-graph #rag"
    tags = _extract_tags(body)
    assert tags.count("rag") == 1
    assert "knowledge-graph" in tags


def test_source_type_book_chapter() -> None:
    p = Path("/Users/wiseer/Dropbox/ai_readings/2.knowledge graphs/ch13.pdf")
    assert _source_type(p) == "book-chapter"


def test_source_type_paper() -> None:
    p = Path("/Users/wiseer/Dropbox/ai_readings/ai_engineering/paper.pdf")
    assert _source_type(p) == "paper"
