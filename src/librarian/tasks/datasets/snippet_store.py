"""Generic snippet store interface for FAQ bypass retrieval.

Defines the ``SnippetStore`` protocol and a ``DuckDBSnippetStore`` adapter
that wraps an existing DuckDB connection for keyword-based lookup.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SnippetStore(Protocol):
    """Protocol for snippet storage backends used in regression eval."""

    def insert(self, snippets: list[dict[str, Any]]) -> None: ...

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]: ...

    def get_by_doc(self, doc_id: str) -> list[dict[str, Any]]: ...


class InMemorySnippetStore:
    """Simple in-memory snippet store for testing."""

    def __init__(self) -> None:
        self._snippets: list[dict[str, Any]] = []

    def insert(self, snippets: list[dict[str, Any]]) -> None:
        self._snippets.extend(snippets)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Naive keyword search: score by number of matching tokens."""
        query_tokens = set(query.lower().split())
        scored = []
        for snippet in self._snippets:
            text_tokens = set(snippet.get("text", "").lower().split())
            overlap = len(query_tokens & text_tokens)
            if overlap > 0:
                scored.append((overlap, snippet))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, snippet in scored[:k]:
            results.append({**snippet, "score": float(score)})
        return results

    def get_by_doc(self, doc_id: str) -> list[dict[str, Any]]:
        return [s for s in self._snippets if s.get("doc_id") == doc_id]
