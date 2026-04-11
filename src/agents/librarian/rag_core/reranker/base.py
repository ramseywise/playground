"""Reranker-layer protocol — re-scoring retrieved chunks."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agents.librarian.rag_core.schemas.chunks import GradedChunk, RankedChunk


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        chunks: list[GradedChunk],
        top_k: int = 3,
    ) -> list[RankedChunk]:
        """Return chunks sorted by relevance_score desc, len <= top_k.

        relevance_score in [0, 1]; rank is 1-indexed.
        """
        ...


__all__ = ["Reranker"]
