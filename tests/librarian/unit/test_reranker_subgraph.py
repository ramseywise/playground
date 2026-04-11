from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestration.nodes.reranker import RerankerSubgraph
from librarian.schemas.chunks import (
    Chunk,
    ChunkMetadata,
    GradedChunk,
    RankedChunk,
)
from librarian.schemas.state import LibrarianState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, text: str = "some text") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(url=f"https://x.com/{chunk_id}", title="T", doc_id="d1"),
    )


def _graded(chunk_id: str, score: float = 0.5, relevant: bool = True) -> GradedChunk:
    return GradedChunk(chunk=_chunk(chunk_id), score=score, relevant=relevant)


def _ranked(chunk_id: str, relevance_score: float, rank: int) -> RankedChunk:
    return RankedChunk(
        chunk=_chunk(chunk_id), relevance_score=relevance_score, rank=rank
    )


def _state(**kwargs: object) -> LibrarianState:
    base: LibrarianState = {"query": "what is auth?", "intent": "lookup"}
    base.update(kwargs)  # type: ignore[typeddict-item]
    return base


def _mock_reranker(return_value: list[RankedChunk]) -> MagicMock:
    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=return_value)
    return reranker


# ---------------------------------------------------------------------------
# No graded_chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_empty_graded_returns_empty() -> None:
    sg = RerankerSubgraph(reranker=_mock_reranker([]), top_k=3)
    result = await sg.run(_state())
    assert result["reranked_chunks"] == []
    assert result["confidence_score"] == 0.0


@pytest.mark.asyncio
async def test_run_empty_graded_reranker_not_called() -> None:
    mock = _mock_reranker([])
    sg = RerankerSubgraph(reranker=mock, top_k=3)
    await sg.run(_state())
    mock.rerank.assert_not_called()


# ---------------------------------------------------------------------------
# Normal path — relevant chunks present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_reranked_chunks() -> None:
    ranked = [_ranked("c1", 0.9, 1), _ranked("c2", 0.7, 2)]
    sg = RerankerSubgraph(reranker=_mock_reranker(ranked), top_k=2)
    result = await sg.run(_state(graded_chunks=[_graded("c1"), _graded("c2")]))
    assert result["reranked_chunks"] == ranked


@pytest.mark.asyncio
async def test_run_confidence_is_max_relevance_score() -> None:
    ranked = [_ranked("c1", 0.9, 1), _ranked("c2", 0.6, 2)]
    sg = RerankerSubgraph(reranker=_mock_reranker(ranked), top_k=2)
    result = await sg.run(_state(graded_chunks=[_graded("c1"), _graded("c2")]))
    assert result["confidence_score"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_run_uses_standalone_query() -> None:
    mock = _mock_reranker([_ranked("c1", 0.8, 1)])
    sg = RerankerSubgraph(reranker=mock, top_k=1)
    await sg.run(
        _state(
            query="original",
            standalone_query="rewritten",
            graded_chunks=[_graded("c1")],
        )
    )
    call_query = mock.rerank.call_args[0][0]
    assert call_query == "rewritten"


@pytest.mark.asyncio
async def test_run_passes_top_k_to_reranker() -> None:
    mock = _mock_reranker([])
    sg = RerankerSubgraph(reranker=mock, top_k=5)
    await sg.run(_state(graded_chunks=[_graded("c1")]))
    call_top_k = mock.rerank.call_args[1].get("top_k") or mock.rerank.call_args[0][2]
    assert call_top_k == 5


# ---------------------------------------------------------------------------
# Relevant-chunk filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_filters_to_relevant_only() -> None:
    mock = _mock_reranker([_ranked("c1", 0.9, 1)])
    sg = RerankerSubgraph(reranker=mock, top_k=3)
    graded = [_graded("c1", relevant=True), _graded("c2", relevant=False)]
    await sg.run(_state(graded_chunks=graded))

    passed_chunks = mock.rerank.call_args[0][1]
    assert all(g.relevant for g in passed_chunks)
    assert len(passed_chunks) == 1


@pytest.mark.asyncio
async def test_run_fallback_all_chunks_when_none_relevant() -> None:
    """When no chunk is relevant, pass all chunks to avoid empty rerank."""
    mock = _mock_reranker([_ranked("c1", 0.3, 1), _ranked("c2", 0.2, 2)])
    sg = RerankerSubgraph(reranker=mock, top_k=3)
    graded = [_graded("c1", relevant=False), _graded("c2", relevant=False)]
    await sg.run(_state(graded_chunks=graded))

    passed_chunks = mock.rerank.call_args[0][1]
    assert len(passed_chunks) == 2


# ---------------------------------------------------------------------------
# Confidence score edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_confidence_zero_when_reranker_returns_empty() -> None:
    sg = RerankerSubgraph(reranker=_mock_reranker([]), top_k=3)
    # graded is non-empty so reranker IS called, but returns empty
    result = await sg.run(_state(graded_chunks=[_graded("c1")]))
    assert result["confidence_score"] == 0.0


@pytest.mark.asyncio
async def test_run_confidence_single_chunk() -> None:
    ranked = [_ranked("c1", 0.55, 1)]
    sg = RerankerSubgraph(reranker=_mock_reranker(ranked), top_k=1)
    result = await sg.run(_state(graded_chunks=[_graded("c1")]))
    assert result["confidence_score"] == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_run_output_keys_always_present() -> None:
    sg = RerankerSubgraph(reranker=_mock_reranker([]), top_k=3)
    for state in [
        _state(),
        _state(graded_chunks=[]),
        _state(graded_chunks=[_graded("c1")]),
    ]:
        result = await sg.run(state)
        assert "reranked_chunks" in result
        assert "confidence_score" in result
