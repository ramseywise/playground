from __future__ import annotations

from typing import TYPE_CHECKING

from agents.librarian.pipeline.schemas.chunks import Chunk, ChunkMetadata
from agents.librarian.pipeline.schemas.retrieval import RetrievalResult
from agents.librarian.utils.logging import get_logger

if TYPE_CHECKING:
    from agents.librarian.tools.storage.snippet_db import SnippetDB

log = get_logger(__name__)


class SnippetRetriever:
    """Retriever protocol implementation backed by SnippetDB (DuckDB FTS/ILIKE).

    Used for the *snippet* retrieval path: fast keyword-based lookup of
    pre-extracted short facts, bypassing embedding and vector search.

    The ``query_vector`` argument is accepted for protocol compatibility but
    is not used — retrieval is keyword-only.
    """

    def __init__(self, snippet_db: SnippetDB) -> None:
        self._db = snippet_db

    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        """Search the snippet store and return up to *k* results."""
        rows = self._db.search_snippets(query_text, k=k)

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
                    chunk=chunk,
                    score=float(row.get("score", 1.0)),
                    source="bm25",
                )
            )

        log.info(
            "snippet_retriever.search",
            query=query_text[:80],
            results=len(results),
        )
        return results

    async def upsert(self, chunks: list[Chunk]) -> None:
        """No-op: snippets are written via IngestionPipeline, not through the Retriever protocol."""
