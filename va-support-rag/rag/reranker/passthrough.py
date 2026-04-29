"""Passthrough reranker — returns chunks in original score order, no model call."""

from __future__ import annotations

from rag.schemas.chunks import GradedChunk, RankedChunk


class PassthroughReranker:
    async def rerank(
        self, query: str, chunks: list[GradedChunk], top_k: int = 3
    ) -> list[RankedChunk]:
        top = sorted(chunks, key=lambda g: g.score, reverse=True)[:top_k]
        return [
            RankedChunk(chunk=g.chunk, relevance_score=g.score, rank=i + 1)
            for i, g in enumerate(top)
        ]
