"""Storage protocols shared across agents.

These define the *contract* for each storage role.  Implementations live
inside the agent that needs them (e.g. ``agents.librarian.storage``).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    """Dense vector storage backend (Chroma, OpenSearch, DuckDB+vss, InMemory)."""

    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict]: ...

    async def upsert(self, items: list[dict]) -> None: ...


@runtime_checkable
class MetadataStore(Protocol):
    """Document-level metadata persistence (DuckDB)."""

    def get(self, doc_id: str) -> dict | None: ...

    def upsert(self, doc_id: str, metadata: dict) -> None: ...


@runtime_checkable
class TraceStore(Protocol):
    """Golden trace / FAQ lookup store (DuckDB FTS / BM25)."""

    def search(self, query: str, k: int = 5) -> list[dict]: ...

    def insert(self, items: list[dict]) -> None: ...


@runtime_checkable
class GraphStore(Protocol):
    """Knowledge graph storage (DuckDB recursive CTEs, future dedicated engine).

    Placeholder — will be fleshed out when graph RAG is implemented.
    """

    def query(self, cypher_or_sql: str, params: dict | None = None) -> list[dict]: ...
