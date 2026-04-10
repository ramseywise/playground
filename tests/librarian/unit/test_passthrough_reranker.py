"""Tests for the passthrough reranker."""

from __future__ import annotations

import pytest

from agents.librarian.reranker.passthrough import PassthroughReranker
from agents.librarian.schemas.chunks import Chunk, ChunkMetadata, GradedChunk


def _graded(id_: str, score: float, *, relevant: bool = True) -> GradedChunk:
    return GradedChunk(
        chunk=Chunk(
            id=id_,
            text="text",
            metadata=ChunkMetadata(url="https://x.com", title="T", doc_id="d"),
        ),
        score=score,
        relevant=relevant,
    )


class TestPassthroughReranker:
    @pytest.mark.asyncio
    async def test_preserves_order_by_score(self) -> None:
        reranker = PassthroughReranker()
        chunks = [_graded("a", 0.3), _graded("b", 0.9), _graded("c", 0.5)]
        result = await reranker.rerank("query", chunks, top_k=3)
        assert [r.chunk.id for r in result] == ["b", "c", "a"]

    @pytest.mark.asyncio
    async def test_respects_top_k(self) -> None:
        reranker = PassthroughReranker()
        chunks = [_graded("a", 0.9), _graded("b", 0.5), _graded("c", 0.3)]
        result = await reranker.rerank("query", chunks, top_k=2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_assigns_ranks(self) -> None:
        reranker = PassthroughReranker()
        chunks = [_graded("a", 0.9), _graded("b", 0.5)]
        result = await reranker.rerank("query", chunks)
        assert result[0].rank == 1
        assert result[1].rank == 2

    @pytest.mark.asyncio
    async def test_clamps_score_to_one(self) -> None:
        reranker = PassthroughReranker()
        chunks = [_graded("a", 1.5)]
        result = await reranker.rerank("query", chunks)
        assert result[0].relevance_score == 1.0

    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        reranker = PassthroughReranker()
        result = await reranker.rerank("query", [])
        assert result == []
