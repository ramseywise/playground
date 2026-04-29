"""Unit tests for :mod:`app.rag.retrieval.pipeline` (no vector index required)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.schemas.chunks import Chunk, ChunkMetadata, GradedChunk, RankedChunk
from rag.retrieval import pipeline as pipeline_mod


def _graded(score: float) -> GradedChunk:
    return GradedChunk(
        chunk=Chunk(id="1", text="x", metadata=ChunkMetadata()),
        score=score,
    )


def _ranked(score: float) -> RankedChunk:
    return RankedChunk(
        chunk=Chunk(id="1", text="x", metadata=ChunkMetadata()),
        relevance_score=score,
        rank=1,
    )


@pytest.mark.asyncio
async def test_retrieve_graded_chunks_delegates_to_ensemble() -> None:
    mock_ens = MagicMock()
    mock_ens.retrieve = AsyncMock(return_value=[_graded(0.5)])

    with patch.object(pipeline_mod, "get_ensemble_retriever", return_value=mock_ens):
        out = await pipeline_mod.retrieve_graded_chunks(["q1", "q2"], k=7)

    assert len(out) == 1
    mock_ens.retrieve.assert_awaited_once_with(["q1", "q2"], k=7)


@pytest.mark.asyncio
async def test_rerank_graded_chunks_delegates_to_reranker() -> None:
    mock_rr = MagicMock()
    mock_rr.rerank = AsyncMock(return_value=[_ranked(0.9)])
    graded = [_graded(0.4)]

    with patch.object(pipeline_mod, "get_reranker", return_value=mock_rr):
        out = await pipeline_mod.rerank_graded_chunks("query", graded, top_k=5)

    assert len(out) == 1
    mock_rr.rerank.assert_awaited_once_with("query", graded, top_k=5)
