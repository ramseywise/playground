"""Retrieval-layer protocols — embedding and vector search contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agents.librarian.pipeline.schemas.chunks import Chunk
from agents.librarian.pipeline.schemas.retrieval import RetrievalResult


@runtime_checkable
class Embedder(Protocol):
    """E5 prefix rule is enforced at this level.

    - embed_query:   prepends "query: "   — for search-time queries
    - embed_passage: prepends "passage: " — for indexing document text
    - embed_passages: batch version of embed_passage
    """

    def embed_query(self, text: str) -> list[float]: ...

    async def aembed_query(self, text: str) -> list[float]: ...

    def embed_passage(self, text: str) -> list[float]: ...

    async def aembed_passage(self, text: str) -> list[float]: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    async def aembed_passages(self, texts: list[str]) -> list[list[float]]: ...


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


__all__ = ["Embedder", "Retriever"]
