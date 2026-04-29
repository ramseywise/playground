"""Reranker protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag.schemas.chunks import GradedChunk, RankedChunk


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        chunks: list[GradedChunk],
        top_k: int = 3,
    ) -> list[RankedChunk]: ...


__all__ = ["Reranker"]
