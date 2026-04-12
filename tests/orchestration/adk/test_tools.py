"""Tests for ADK tool functions wrapping the Librarian retrieval stack."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from librarian.schemas.chunks import Chunk, ChunkMetadata, GradedChunk, RankedChunk
from librarian.schemas.retrieval import RetrievalResult
from orchestration.adk.tools import (
    _check_configured,
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


def test_check_configured_raises_when_not_configured() -> None:
    """Tools must raise if configure_tools() was never called."""
    import orchestration.adk.tools as tools_mod

    # Reset module state
    tools_mod._retriever = None
    tools_mod._embedder = None
    tools_mod._reranker = None

    with pytest.raises(RuntimeError, match="not configured"):
        _check_configured()


def test_configure_tools_sets_components() -> None:
    retriever = MagicMock()
    embedder = MagicMock()
    reranker = MagicMock()
    configure_tools(retriever=retriever, embedder=embedder, reranker=reranker)

    r, e, rr = _check_configured()
    assert r is retriever
    assert e is embedder
    assert rr is reranker


# ---------------------------------------------------------------------------
# search_knowledge_base
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_knowledge_base_returns_results() -> None:
    """search_knowledge_base should embed the query, search, and return formatted results."""
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
    assert result["results"][0]["text"] == "First passage about auth."

    mock_embedder.aembed_query.assert_awaited_once_with("what is authentication?")
    mock_retriever.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_knowledge_base_passes_k() -> None:
    """num_results should be forwarded as k to the retriever."""
    mock_embedder = MagicMock()
    mock_embedder.aembed_query = AsyncMock(return_value=[0.1] * 64)

    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])

    configure_tools(
        retriever=mock_retriever,
        embedder=mock_embedder,
        reranker=MagicMock(),
    )

    await search_knowledge_base("test query", num_results=20)

    call_kwargs = mock_retriever.search.call_args
    assert call_kwargs.kwargs["k"] == 20


# ---------------------------------------------------------------------------
# rerank_results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerank_results_returns_ranked() -> None:
    """rerank_results should convert passages to GradedChunks and rerank."""
    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(
        return_value=[
            _make_ranked_chunk("c2", "Billing info", 0.92, 1),
            _make_ranked_chunk("c1", "Auth info", 0.71, 2),
        ]
    )

    configure_tools(
        retriever=MagicMock(),
        embedder=MagicMock(),
        reranker=mock_reranker,
    )

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
    assert result["results"][0]["relevance_score"] == 0.92
    assert result["confidence"] == 0.92

    mock_reranker.rerank.assert_awaited_once()
    call_args = mock_reranker.rerank.call_args
    assert call_args.kwargs["top_k"] == 2


@pytest.mark.asyncio
async def test_rerank_results_empty_returns_zero_confidence() -> None:
    """Empty reranker output should give confidence 0.0."""
    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=[])

    configure_tools(
        retriever=MagicMock(),
        embedder=MagicMock(),
        reranker=mock_reranker,
    )

    result = await rerank_results("test", [], top_k=3)

    assert result["results"] == []
    assert result["confidence"] == 0.0
