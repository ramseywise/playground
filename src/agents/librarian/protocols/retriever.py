"""Retriever protocol — vector + text search."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agents.librarian.schemas.chunks import Chunk
from agents.librarian.schemas.retrieval import RetrievalResult


@runtime_checkable
class Retriever(Protocol):
    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        k: int = 10,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]: ...

    async def upsert(self, chunks: list[Chunk]) -> None: ...
