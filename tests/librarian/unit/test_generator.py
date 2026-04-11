from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from librarian.generation.generator import (
    build_prompt,
    call_llm,
    extract_citations,
)
from librarian.generation.prompts import SYSTEM_PROMPTS, get_system_prompt
from librarian.schemas.chunks import Chunk, ChunkMetadata, RankedChunk
from librarian.schemas.state import LibrarianState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ranked(
    url: str, title: str, text: str, rank: int, score: float = 0.8
) -> RankedChunk:
    return RankedChunk(
        chunk=Chunk(
            id=f"c{rank}",
            text=text,
            metadata=ChunkMetadata(url=url, title=title, doc_id="d1"),
        ),
        relevance_score=score,
        rank=rank,
    )


def _state(**kwargs: object) -> LibrarianState:
    base: LibrarianState = {"query": "what is X?", "intent": "lookup"}
    base.update(kwargs)  # type: ignore[typeddict-item]
    return base


@dataclass
class _Msg:
    type: str
    content: str


# ---------------------------------------------------------------------------
# prompts.py
# ---------------------------------------------------------------------------


def test_all_intents_have_prompts() -> None:
    for intent in ("lookup", "explore", "compare", "conversational", "out_of_scope"):
        assert intent in SYSTEM_PROMPTS
        assert len(SYSTEM_PROMPTS[intent]) > 10


def test_get_system_prompt_known_intent() -> None:
    assert get_system_prompt("lookup") == SYSTEM_PROMPTS["lookup"]


def test_get_system_prompt_unknown_falls_back_to_lookup() -> None:
    assert get_system_prompt("unknown_intent") == SYSTEM_PROMPTS["lookup"]


def test_get_system_prompt_case_insensitive() -> None:
    assert get_system_prompt("LOOKUP") == SYSTEM_PROMPTS["lookup"]


# ---------------------------------------------------------------------------
# build_prompt — direct intents (no context)
# ---------------------------------------------------------------------------


def test_build_prompt_conversational_no_context() -> None:
    chunks = [_ranked("https://x.com", "T", "some text", 1)]
    system, messages = build_prompt(_state(intent="conversational"), chunks)
    assert system == SYSTEM_PROMPTS["conversational"]
    # No grounded message injected for conversational
    assert not any("Use the following sources" in m["content"] for m in messages)


def test_build_prompt_out_of_scope_no_context() -> None:
    chunks = [_ranked("https://x.com", "T", "text", 1)]
    system, messages = build_prompt(_state(intent="out_of_scope"), chunks)
    assert system == SYSTEM_PROMPTS["out_of_scope"]


def test_build_prompt_empty_chunks_no_context() -> None:
    system, messages = build_prompt(_state(intent="lookup"), [])
    assert "Use the following sources" not in str(messages)


# ---------------------------------------------------------------------------
# build_prompt — retrieval intents (context injected)
# ---------------------------------------------------------------------------


def test_build_prompt_lookup_injects_context() -> None:
    chunks = [
        _ranked(
            "https://docs.example.com/auth",
            "Auth Guide",
            "API keys expire after 24h",
            1,
        )
    ]
    system, messages = build_prompt(
        _state(intent="lookup", query="how do API keys work?"), chunks
    )
    assert system == SYSTEM_PROMPTS["lookup"]
    last = messages[-1]
    assert "API keys expire after 24h" in last["content"]
    assert "https://docs.example.com/auth" in last["content"]


def test_build_prompt_uses_standalone_query_over_query() -> None:
    chunks = [_ranked("https://x.com", "T", "text", 1)]
    _, messages = build_prompt(
        _state(query="original", standalone_query="rewritten standalone"), chunks
    )
    assert "rewritten standalone" in messages[-1]["content"]


def test_build_prompt_multiple_chunks_joined_with_separator() -> None:
    chunks = [
        _ranked("https://a.com", "A", "chunk A text", 1),
        _ranked("https://b.com", "B", "chunk B text", 2),
    ]
    _, messages = build_prompt(_state(intent="lookup"), chunks)
    assert "---" in messages[-1]["content"]
    assert "chunk A text" in messages[-1]["content"]
    assert "chunk B text" in messages[-1]["content"]


def test_build_prompt_preserves_conversation_history() -> None:
    history = [_Msg(type="human", content="hi"), _Msg(type="ai", content="hello")]
    chunks = [_ranked("https://x.com", "T", "text", 1)]
    state = _state(intent="lookup", messages=history)
    _, messages = build_prompt(state, chunks)
    # History preserved (converted to dicts); last message is the grounded question
    assert messages[0]["content"] == "hi"
    assert messages[1]["content"] == "hello"
    assert "Use the following sources" in messages[-1]["content"]


def test_build_prompt_replaces_trailing_human_message() -> None:
    history = [_Msg(type="human", content="original question")]
    chunks = [_ranked("https://x.com", "T", "text", 1)]
    state = _state(intent="lookup", messages=history)
    _, messages = build_prompt(state, chunks)
    # Original HumanMessage replaced by grounded version
    assert "original question" not in messages[-1]["content"]
    assert "Use the following sources" in messages[-1]["content"]


# ---------------------------------------------------------------------------
# call_llm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_llm_returns_content(mock_llm: MagicMock) -> None:
    mock_llm.generate = AsyncMock(return_value="42 is the answer")
    result = await call_llm(
        mock_llm, "you are helpful", [{"role": "user", "content": "what?"}]
    )
    assert result == "42 is the answer"


@pytest.mark.asyncio
async def test_call_llm_passes_system_and_messages(mock_llm: MagicMock) -> None:
    mock_llm.generate = AsyncMock(return_value="ok")
    await call_llm(
        mock_llm,
        "system instructions",
        [{"role": "user", "content": "hello"}],
    )
    # Verify generate was called with system as first arg, messages as second
    call_args = mock_llm.generate.call_args
    assert call_args[0][0] == "system instructions"
    assert call_args[0][1] == [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# extract_citations
# ---------------------------------------------------------------------------


def test_extract_citations_deduplicates_by_url() -> None:
    chunks = [
        _ranked("https://a.com", "A", "text", 1),
        _ranked("https://b.com", "B", "text", 2),
        _ranked("https://a.com", "A duplicate", "text", 3),
    ]
    citations = extract_citations(chunks)
    urls = [c["url"] for c in citations]
    assert urls == ["https://a.com", "https://b.com"]


def test_extract_citations_preserves_rank_order() -> None:
    chunks = [
        _ranked("https://b.com", "B", "text", 1),
        _ranked("https://a.com", "A", "text", 2),
    ]
    citations = extract_citations(chunks)
    assert citations[0]["url"] == "https://b.com"
    assert citations[1]["url"] == "https://a.com"


def test_extract_citations_empty() -> None:
    assert extract_citations([]) == []


def test_extract_citations_structure() -> None:
    chunks = [_ranked("https://x.com/doc", "Doc Title", "text", 1)]
    citations = extract_citations(chunks)
    assert citations[0] == {"url": "https://x.com/doc", "title": "Doc Title"}
