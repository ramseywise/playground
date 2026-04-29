"""SnippetRetriever — keyword-based lookup backed by DuckDB FTS/ILIKE.

Bypasses embedding and vector search for fast factual snippet retrieval.
query_vector is accepted for protocol compatibility but is unused.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

from rag.schemas.chunks import Chunk, ChunkMetadata
from rag.schemas.retrieval import RetrievalResult

log = logging.getLogger(__name__)


class SnippetDB(Protocol):
    """Duck-typed snippet index (e.g. DuckDB FTS); no ``storage.snippet_db`` in-tree."""

    def search_snippets(self, query_text: str, k: int = 5) -> list[dict[str, Any]]: ...
    def insert_snippets(self, records: list[dict[str, Any]]) -> None: ...


class MetadataDB(Protocol):
    """Duck-typed document metadata store; no ``storage.metadata_db`` in-tree."""

    def document_exists_by_checksum(self, checksum: str) -> bool: ...
    def insert_document(self, doc_id: str, **kwargs: Any) -> None: ...


class SnippetRetriever:
    def __init__(self, snippet_db: SnippetDB) -> None:
        self._db = snippet_db

    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        rows = await asyncio.to_thread(self._db.search_snippets, query_text, k=k)

        results: list[RetrievalResult] = []
        for row in rows:
            chunk = Chunk(
                id=row["id"],
                text=row["text"],
                metadata=ChunkMetadata(
                    url="",
                    title=row.get("title", ""),
                    doc_id=row["doc_id"],
                    topic=row.get("topic"),
                ),
            )
            results.append(
                RetrievalResult(
                    chunk=chunk, score=float(row.get("score", 1.0)), source="bm25"
                )
            )

        log.info(
            "snippet_retriever.search query=%s results=%d",
            query_text[:80],
            len(results),
        )
        return results

    async def upsert(self, chunks: list[Chunk]) -> None:
        """No-op: snippets are written via IngestionPipeline."""
