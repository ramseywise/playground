"""Tests for ADK tool functions wrapping the Librarian retrieval stack."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from librarian.schemas.chunks import Chunk, ChunkMetadata, GradedChunk, RankedChunk
from librarian.schemas.retrieval import RetrievalResult
from orchestration.adk.tools import (
    ToolDeps,
    _get_deps,
    analyze_query,
    condense_query,
    configure_tools,
    rerank_results,
    search_knowledge_base,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, text: str, url: str = "https://example.com") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(url=url, title=f"Doc {chunk_id}", doc_id=chunk_id),
    )


def _make_retrieval_result(chunk_id: str, text: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk=_make_chunk(chunk_id, text),
        score=score,
        source="hybrid",
    )


def _make_ranked_chunk(
    chunk_id: str, text: str, score: float, rank: int
) -> RankedChunk:
    return RankedChunk(
        chunk=_make_chunk(chunk_id, text),
        relevance_score=score,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# configure_tools
# ---------------------------------------------------------------------------


def test_get_deps_raises_when_not_configured() -> None:
    """Tools must raise if configure_tools() was never called."""
    import orchestration.adk.tools as tools_mod

    tools_mod._deps = None
    with pytest.raises(RuntimeError, match="not configured"):
        _get_deps()


def test_configure_tools_returns_deps() -> None:
    deps = configure_tools(
        retriever=MagicMock(),
        embedder=MagicMock(),
        reranker=MagicMock(),
    )
    assert isinstance(deps, ToolDeps)
    assert deps.retriever is not None
    assert deps.analyzer is not None  # auto-created


def test_configure_tools_accepts_condenser_llm() -> None:
    mock_llm = MagicMock()
    deps = configure_tools(
        retriever=MagicMock(),
        embedder=MagicMock(),
        reranker=MagicMock(),
        condenser_llm=mock_llm,
    )
    assert deps.condenser_llm is mock_llm


# ---------------------------------------------------------------------------
# search_knowledge_base
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_knowledge_base_returns_results() -> None:
    mock_embedder = MagicMock()
    mock_embedder.aembed_query = AsyncMock(return_value=[0.1] * 64)

    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(
        return_value=[
            _make_retrieval_result("c1", "First passage about auth.", 0.95),
            _make_retrieval_result("c2", "Second passage about billing.", 0.72),
        ]
    )

    configure_tools(
        retriever=mock_retriever,
        embedder=mock_embedder,
        reranker=MagicMock(),
    )

    result = await search_knowledge_base("what is authentication?", num_results=5)

    assert result["total"] == 2
    assert len(result["results"]) == 2
    assert result["results"][0]["chunk_id"] == "c1"
    assert result["results"][0]["score"] == 0.95

    mock_embedder.aembed_query.assert_awaited_once_with("what is authentication?")
    mock_retriever.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_knowledge_base_passes_k() -> None:
    mock_embedder = MagicMock()
    mock_embedder.aembed_query = AsyncMock(return_value=[0.1] * 64)
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])

    configure_tools(
        retriever=mock_retriever, embedder=mock_embedder, reranker=MagicMock()
    )

    await search_knowledge_base("test query", num_results=20)
    assert mock_retriever.search.call_args.kwargs["k"] == 20


# ---------------------------------------------------------------------------
# rerank_results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerank_results_returns_ranked() -> None:
    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(
        return_value=[
            _make_ranked_chunk("c2", "Billing info", 0.92, 1),
            _make_ranked_chunk("c1", "Auth info", 0.71, 2),
        ]
    )

    configure_tools(retriever=MagicMock(), embedder=MagicMock(), reranker=mock_reranker)

    passages = [
        {
            "text": "Auth info",
            "chunk_id": "c1",
            "url": "https://a.com",
            "title": "A",
            "score": "0.8",
        },
        {
            "text": "Billing info",
            "chunk_id": "c2",
            "url": "https://b.com",
            "title": "B",
            "score": "0.7",
        },
    ]

    result = await rerank_results("what is billing?", passages, top_k=2)

    assert len(result["results"]) == 2
    assert result["results"][0]["rank"] == 1
    assert result["confidence"] == 0.92
    mock_reranker.rerank.assert_awaited_once()


@pytest.mark.asyncio
async def test_rerank_results_empty_returns_zero_confidence() -> None:
    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=[])

    configure_tools(retriever=MagicMock(), embedder=MagicMock(), reranker=mock_reranker)

    result = await rerank_results("test", [], top_k=3)
    assert result["results"] == []
    assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# analyze_query
# ---------------------------------------------------------------------------


def test_analyze_query_returns_intent_and_entities() -> None:
    configure_tools(retriever=MagicMock(), embedder=MagicMock(), reranker=MagicMock())

    result = analyze_query("how does OAuth2 compare to SAML for authentication?")

    assert "intent" in result
    assert "entities" in result
    assert "complexity" in result
    assert "expanded_terms" in result
    assert "retrieval_mode" in result
    assert result["confidence"] > 0


def test_analyze_query_simple_query() -> None:
    configure_tools(retriever=MagicMock(), embedder=MagicMock(), reranker=MagicMock())

    result = analyze_query("what is OAuth?")
    assert result["complexity"] in ("simple", "moderate")


# ---------------------------------------------------------------------------
# condense_query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_condense_query_single_turn_passes_through() -> None:
    """Single-turn queries should not be rewritten."""
    configure_tools(retriever=MagicMock(), embedder=MagicMock(), reranker=MagicMock())

    result = await condense_query("what is OAuth?", [])
    assert result["standalone_query"] == "what is OAuth?"
    assert result["was_rewritten"] is False


@pytest.mark.asyncio
async def test_condense_query_no_llm_passes_through() -> None:
    """Without condenser_llm configured, should pass through."""
    configure_tools(
        retriever=MagicMock(),
        embedder=MagicMock(),
        reranker=MagicMock(),
        condenser_llm=None,
    )

    history = [
        {"role": "user", "content": "what is OAuth?"},
        {"role": "assistant", "content": "OAuth is a protocol..."},
        {"role": "user", "content": "how about SAML?"},
    ]
    result = await condense_query("how about SAML?", history)
    assert result["standalone_query"] == "how about SAML?"
    assert result["was_rewritten"] is False


@pytest.mark.asyncio
async def test_condense_query_rewrites_with_llm() -> None:
    """With condenser_llm, should rewrite the query."""
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value="How does SAML compare to OAuth for authentication?"
    )

    configure_tools(
        retriever=MagicMock(),
        embedder=MagicMock(),
        reranker=MagicMock(),
        condenser_llm=mock_llm,
    )

    history = [
        {"role": "user", "content": "what is OAuth?"},
        {"role": "assistant", "content": "OAuth is a protocol..."},
        {"role": "user", "content": "how about SAML?"},
    ]
    result = await condense_query("how about SAML?", history)
    assert (
        result["standalone_query"]
        == "How does SAML compare to OAuth for authentication?"
    )
    assert result["was_rewritten"] is True
    mock_llm.generate.assert_awaited_once()
