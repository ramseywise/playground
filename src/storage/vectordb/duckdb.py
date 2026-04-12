"""DuckDB vector retrieval backend.

Uses DuckDB's built-in array_cosine_similarity for brute-force cosine search.
No vss extension required. Stores chunks in a ``rag_chunks`` table.

When to use:
  - Local dev / CI with no Docker dependency
  - Small corpora (<50k chunks) where linear scan is acceptable
  - When you already have a DuckDB database and want to co-locate RAG data

Trade-offs vs ChromaDB:
  - No HNSW index → O(n) scan per query (slower at scale)
  - SQL-native → easy joins with other app tables (e.g. user metadata)
  - No sparse/BM25 signal — vector-only search (add term_overlap in Python if needed)
  - Shares DuckDB's single-writer lock with the rest of the application

Implements the Retriever Protocol from retrieval/base.py so it is a drop-in
replacement for InMemoryRetriever, ChromaRetriever, or OpenSearchRetriever.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from core.retrieval.rrf import fuse_rankings
from core.retrieval.scoring import term_overlap
from core.schemas.chunks import Chunk, ChunkMetadata, GradedChunk
from core.schemas.retrieval import RetrievalResult
from core.logging import get_logger

log = get_logger(__name__)

_DEFAULT_DB_PATH = ".duckdb/librarian.db"
_EMBEDDING_DIMS = 1024  # multilingual-e5-large default; override via constructor


class DuckDBRetriever:
    """DuckDB-backed retriever using array_cosine_similarity for vector search.

    Hybrid score uses Reciprocal Rank Fusion over vector and keyword rankings.
    Matches the InMemoryRetriever and ChromaRetriever interface for test/prod parity.

    Args:
        db_path:         Path to the DuckDB file. Created on first use.
        table_name:      Table name for chunk storage.
        embedding_dims:  Vector dimension — must match the embedder used at ingest time.
        bm25_weight:     Weight for term-overlap signal.
        vector_weight:   Weight for cosine similarity signal.
    """

    def __init__(
        self,
        db_path: str | Path = _DEFAULT_DB_PATH,
        table_name: str = "rag_chunks",
        embedding_dims: int = _EMBEDDING_DIMS,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> None:
        self._db_path = str(db_path)
        self._table_name = table_name
        self._embedding_dims = embedding_dims
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    def _connect(self) -> Any:
        """Open a fresh DuckDB connection. Caller must close it."""
        import duckdb  # type: ignore[import-untyped]

        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(self._db_path)

    def _ensure_table(self, conn: Any) -> None:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                chunk_id    VARCHAR PRIMARY KEY,
                text        VARCHAR NOT NULL,
                url         VARCHAR DEFAULT '',
                title       VARCHAR DEFAULT '',
                section     VARCHAR DEFAULT '',
                doc_id      VARCHAR DEFAULT '',
                language    VARCHAR DEFAULT 'en',
                namespace   VARCHAR,
                topic       VARCHAR,
                parent_id   VARCHAR,
                embedding   FLOAT[{self._embedding_dims}],
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    async def upsert(self, chunks: list[Chunk]) -> None:
        """Upsert chunks with precomputed embeddings into the DuckDB table.

        Chunks without embeddings are skipped with a warning.
        Uses DELETE + INSERT per chunk (DuckDB has no native UPSERT with array cols).
        """
        if not chunks:
            return
        await asyncio.to_thread(self._upsert_sync, chunks)

    def _upsert_sync(self, chunks: list[Chunk]) -> None:
        conn = self._connect()
        try:
            self._ensure_table(conn)
            indexed = 0
            for chunk in chunks:
                if chunk.embedding is None:
                    log.warning("duckdb.upsert.missing_embedding", chunk_id=chunk.id)
                    continue
                meta = chunk.metadata
                conn.execute(
                    f"DELETE FROM {self._table_name} WHERE chunk_id = ?",
                    [chunk.id],
                )
                conn.execute(
                    f"""
                    INSERT INTO {self._table_name}
                        (chunk_id, text, url, title, section, doc_id, language,
                         namespace, topic, parent_id, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::FLOAT[{self._embedding_dims}])
                    """,
                    [
                        chunk.id,
                        chunk.text,
                        meta.url,
                        meta.title or "",
                        meta.section or "",
                        meta.doc_id or "",
                        meta.language or "en",
                        meta.namespace,
                        meta.topic,
                        meta.parent_id,
                        chunk.embedding,
                    ],
                )
                indexed += 1
            log.info("duckdb.upsert.done", n=indexed, table=self._table_name)
        finally:
            conn.close()

    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        k: int = 10,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        """Hybrid vector + term-overlap search.

        Fetches top ``k * 3`` candidates by cosine similarity, then re-ranks
        with hybrid score (vector + term_overlap) and returns top ``k``.

        Args:
            query_text:      Raw query string (used for term-overlap scoring).
            query_vector:    Query embedding — must match embedding_dims.
            k:               Number of results to return.
            metadata_filter: Optional equality filters (e.g. {"namespace": "docs"}).

        Returns:
            List of RetrievalResult sorted by hybrid score descending.
        """
        return await asyncio.to_thread(self._search_sync, query_text, query_vector, k, metadata_filter)

    def _search_sync(
        self,
        query_text: str,
        query_vector: list[float],
        k: int = 10,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        conn = self._connect()
        try:
            self._ensure_table(conn)

            _ALLOWED_FILTER_COLS = frozenset(
                {"url", "title", "section", "doc_id", "language", "namespace", "topic", "parent_id"}
            )
            where_clause = ""
            filter_values: list[Any] = []

            if metadata_filter:
                bad = set(metadata_filter) - _ALLOWED_FILTER_COLS
                if bad:
                    raise ValueError(f"Invalid metadata filter keys: {bad}")
                conditions = [f"{col} = ?" for col in metadata_filter]
                where_clause = " AND " + " AND ".join(conditions)
                filter_values = list(metadata_filter.values())

            rows = conn.execute(
                f"""
                SELECT chunk_id, text, url, title, section, doc_id,
                       language, namespace, topic, parent_id,
                       array_cosine_similarity(embedding, ?::FLOAT[{self._embedding_dims}]) AS vec_score
                FROM {self._table_name}
                WHERE embedding IS NOT NULL
                {where_clause}
                ORDER BY vec_score DESC
                LIMIT ?
                """,
                [query_vector] + filter_values + [k * 3],
            ).fetchall()

            vector_rank: list[GradedChunk] = []
            keyword_rank: list[GradedChunk] = []
            for row in rows:
                (
                    chunk_id,
                    text,
                    url,
                    title,
                    section,
                    doc_id,
                    language,
                    namespace,
                    topic,
                    parent_id,
                    vec_score,
                ) = row
                kw_score = term_overlap(query_text, text)
                chunk = Chunk(
                    id=chunk_id,
                    text=text,
                    metadata=ChunkMetadata(
                        url=url,
                        title=title,
                        section=section,
                        doc_id=doc_id,
                        language=language,
                        namespace=namespace,
                        topic=topic,
                        parent_id=parent_id,
                    ),
                )
                vector_rank.append(
                    GradedChunk(
                        chunk=chunk, score=float(vec_score or 0.0), relevant=True
                    )
                )
                keyword_rank.append(
                    GradedChunk(chunk=chunk, score=kw_score, relevant=True)
                )

            fused = fuse_rankings([keyword_rank, vector_rank], top_k=k)
            results = [
                RetrievalResult(chunk=item.chunk, score=item.score, source="hybrid")
                for item in fused
            ]
            log.info("duckdb.search.done", n_results=len(results))
            return results
        finally:
            conn.close()
