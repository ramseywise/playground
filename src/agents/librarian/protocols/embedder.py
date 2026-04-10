"""Embedder protocol — text -> vector."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


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
