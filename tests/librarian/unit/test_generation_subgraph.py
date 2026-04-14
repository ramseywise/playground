from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestration.langgraph.nodes.generation import GeneratorAgent

# Backward-compatible alias used in test names below
GenerationSubgraph = GeneratorAgent
from librarian.schemas.chunks import Chunk, ChunkMetadata, RankedChunk
from librarian.schemas.state import LibrarianState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ranked(chunk_id: str, url: str, title: str, text: str, rank: int) -> RankedChunk:
    return RankedChunk(
        chunk=Chunk(
            id=chunk_id,
            text=text,
            metadata=ChunkMetadata(url=url, title=title, doc_id="d1"),
        ),
        relevance_score=0.8,
        rank=rank,
    )


def _state(**kwargs: object) -> LibrarianState:
    base: LibrarianState = {"query": "what is auth?", "intent": "lookup"}
    base.update(kwargs)  # type: ignore[typeddict-item]
    return base


def _subgraph(
    response: str = "the answer", threshold: float = 0.4
) -> GenerationSubgraph:
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=response)
    return GenerationSubgraph(llm=llm, confidence_threshold=threshold)


# ---------------------------------------------------------------------------
# GenerationSubgraph.run — output keys
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_response_and_citations() -> None:
    sg = _subgraph()
    result = await sg.run(_state())
    assert "response" in result
    assert "citations" in result


@pytest.mark.asyncio
async def test_run_response_is_llm_content() -> None:
    sg = _subgraph(response="42 is the answer")
    result = await sg.run(_state())
    assert result["response"] == "42 is the answer"


@pytest.mark.asyncio
async def test_run_citations_from_reranked_chunks() -> None:
    chunks = [
        _ranked("c1", "https://a.com", "A", "text a", 1),
        _ranked("c2", "https://b.com", "B", "text b", 2),
    ]
    sg = _subgraph()
    result = await sg.run(_state(reranked_chunks=chunks))
    urls = [c["url"] for c in result["citations"]]
    assert "https://a.com" in urls
    assert "https://b.com" in urls


@pytest.mark.asyncio
async def test_run_no_reranked_chunks_empty_citations() -> None:
    sg = _subgraph()
    result = await sg.run(_state())
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_run_deduplicates_citations() -> None:
    chunks = [
        _ranked("c1", "https://a.com", "A", "text a", 1),
        _ranked("c2", "https://a.com", "A dup", "text b", 2),
    ]
    sg = _subgraph()
    result = await sg.run(_state(reranked_chunks=chunks))
    assert len(result["citations"]) == 1


# ---------------------------------------------------------------------------
# GenerationSubgraph.run — prompt routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_conversational_no_context_injected(mock_llm: MagicMock) -> None:
    mock_llm.generate = AsyncMock(return_value="hi there")
    sg = GenerationSubgraph(llm=mock_llm)
    chunks = [_ranked("c1", "https://x.com", "T", "some text", 1)]
    await sg.run(_state(intent="conversational", reranked_chunks=chunks))
    # Verify generate was called — check messages arg for no context injection
    call_args = mock_llm.generate.call_args
    messages = call_args[0][1]  # second positional arg
    content_str = " ".join(m["content"] for m in messages)
    assert "Use the following sources" not in content_str


@pytest.mark.asyncio
async def test_run_lookup_injects_context(mock_llm: MagicMock) -> None:
    mock_llm.generate = AsyncMock(return_value="answer")
    sg = GenerationSubgraph(llm=mock_llm)
    chunks = [_ranked("c1", "https://docs.com", "Docs", "API keys expire after 24h", 1)]
    await sg.run(_state(intent="lookup", reranked_chunks=chunks))
    call_args = mock_llm.generate.call_args
    messages = call_args[0][1]  # second positional arg
    content_str = " ".join(m["content"] for m in messages)
    assert "Use the following sources" in content_str
    assert "API keys expire after 24h" in content_str


# ---------------------------------------------------------------------------
# confidence_gate
# ---------------------------------------------------------------------------


def test_confidence_gate_confident_above_threshold() -> None:
    sg = GenerationSubgraph(llm=MagicMock(), confidence_threshold=0.3)
    result = sg.confidence_gate(_state(confidence_score=0.8))
    assert result["confident"] is True
    assert result["fallback_requested"] is False


def test_confidence_gate_not_confident_below_threshold() -> None:
    sg = GenerationSubgraph(llm=MagicMock(), confidence_threshold=0.3)
    result = sg.confidence_gate(_state(confidence_score=0.1))
    assert result["confident"] is False
    assert result["fallback_requested"] is True


def test_confidence_gate_at_threshold_is_confident() -> None:
    sg = GenerationSubgraph(llm=MagicMock(), confidence_threshold=0.3)
    result = sg.confidence_gate(_state(confidence_score=0.3))
    assert result["confident"] is True


def test_confidence_gate_missing_score_defaults_to_zero() -> None:
    sg = GenerationSubgraph(llm=MagicMock(), confidence_threshold=0.3)
    result = sg.confidence_gate(_state())  # no confidence_score key
    assert result["confident"] is False
    assert result["fallback_requested"] is True


def test_confidence_gate_custom_threshold() -> None:
    sg = GenerationSubgraph(llm=MagicMock(), confidence_threshold=0.7)
    assert sg.confidence_gate(_state(confidence_score=0.5))["confident"] is False
    assert sg.confidence_gate(_state(confidence_score=0.9))["confident"] is True


def test_confidence_gate_both_keys_always_present() -> None:
    sg = GenerationSubgraph(llm=MagicMock())
    for score in [0.0, 0.3, 0.99]:
        result = sg.confidence_gate(_state(confidence_score=score))
        assert "confident" in result
        assert "fallback_requested" in result


def test_default_confidence_threshold_matches_config() -> None:
    from librarian.config import settings

    agent = GeneratorAgent(llm=MagicMock())
    assert agent._threshold == settings.confidence_threshold
