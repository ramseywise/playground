"""Protocols for the snippet DB pipeline."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agents.librarian.snippet_db.models import QAPair, Snippet


@runtime_checkable
class SnippetStore(Protocol):
    """Storage backend for snippets and QA pairs."""

    async def upsert_snippets(self, snippets: list[Snippet]) -> None: ...

    async def upsert_qa_pairs(self, pairs: list[QAPair]) -> None: ...

    async def search_qa(self, query: str, k: int = 5) -> list[QAPair]: ...

    async def get_snippet(self, snippet_id: str) -> Snippet | None: ...


@runtime_checkable
class QAPairGenerator(Protocol):
    """Generates grounded QA pairs from snippets."""

    async def generate(self, snippets: list[Snippet]) -> list[QAPair]: ...
