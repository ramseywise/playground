"""DuckDB-backed document metadata and snippet tables (same file as :class:`DuckDBVectorIndex`).

Tables ``ingest_documents`` and ``ingest_snippets`` live alongside ``rag_chunks`` so one
``rag_index.duckdb`` file holds vectors + ingestion bookkeeping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class DuckDBDocumentMetadataStore:
    """Implements :class:`~app.rag.retrieval.snippet.MetadataDB`."""

    def __init__(self, database_path: Path) -> None:
        import duckdb  # type: ignore[import-untyped]

        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = database_path
        self._con = duckdb.connect(str(database_path))
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS ingest_documents (
                doc_id VARCHAR PRIMARY KEY,
                checksum VARCHAR UNIQUE NOT NULL,
                title VARCHAR,
                source VARCHAR,
                source_file VARCHAR,
                content_type VARCHAR,
                topic VARCHAR,
                word_count INTEGER,
                chunk_count INTEGER,
                snippet_count INTEGER,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def document_exists_by_checksum(self, checksum: str) -> bool:
        row = self._con.execute(
            "SELECT 1 FROM ingest_documents WHERE checksum = ? LIMIT 1",
            [checksum],
        ).fetchone()
        return row is not None

    def insert_document(self, doc_id: str, **kwargs: Any) -> None:
        # One row per doc_id; avoid INSERT OR REPLACE with multiple UNIQUE cols (DuckDB).
        self._con.execute("DELETE FROM ingest_documents WHERE doc_id = ?", [doc_id])
        self._con.execute(
            """
            INSERT INTO ingest_documents
                (doc_id, checksum, title, source, source_file, content_type, topic,
                 word_count, chunk_count, snippet_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                doc_id,
                str(kwargs.get("checksum", "")),
                str(kwargs.get("title", "")),
                str(kwargs.get("source", "")),
                str(kwargs.get("source_file", "")),
                str(kwargs.get("content_type", "")),
                str(kwargs.get("topic", "")),
                int(kwargs.get("word_count") or 0),
                int(kwargs.get("chunk_count") or 0),
                int(kwargs.get("snippet_count") or 0),
            ],
        )
        log.debug("ingest_documents.upsert doc_id=%s", doc_id)


class DuckDBSnippetStore:
    """Implements :class:`~app.rag.retrieval.snippet.SnippetDB` (ILIKE search)."""

    def __init__(self, database_path: Path) -> None:
        import duckdb  # type: ignore[import-untyped]

        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(database_path))
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS ingest_snippets (
                id VARCHAR PRIMARY KEY,
                doc_id VARCHAR,
                text VARCHAR,
                title VARCHAR,
                topic VARCHAR,
                position INTEGER,
                source VARCHAR
            )
        """)

    def insert_snippets(self, records: list[dict[str, Any]]) -> None:
        for r in records:
            self._con.execute(
                """
                INSERT OR REPLACE INTO ingest_snippets
                    (id, doc_id, text, title, topic, position, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(r["id"]),
                    str(r["doc_id"]),
                    str(r["text"]),
                    str(r.get("title", "")),
                    str(r.get("topic", "")),
                    int(r.get("position", 0)),
                    str(r.get("source", "")),
                ],
            )
        log.debug("ingest_snippets.insert n=%d", len(records))

    def search_snippets(self, query_text: str, k: int = 5) -> list[dict[str, Any]]:
        q = query_text.strip()
        if not q:
            return []
        pattern = f"%{q}%"
        rows = self._con.execute(
            """
            SELECT id, doc_id, text, title, topic, position, source
            FROM ingest_snippets
            WHERE text ILIKE ?
            LIMIT ?
            """,
            [pattern, k],
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "id": row[0],
                    "doc_id": row[1],
                    "text": row[2],
                    "title": row[3] or "",
                    "topic": row[4] or "",
                    "position": row[5],
                    "source": row[6] or "",
                    "score": 1.0,
                },
            )
        return out


__all__ = ["DuckDBDocumentMetadataStore", "DuckDBSnippetStore"]
