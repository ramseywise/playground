from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.logging import get_logger

log = get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS snippets (
    id         TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    text       TEXT NOT NULL,
    title      TEXT,
    topic      TEXT,
    position   INTEGER,
    source     TEXT,
    created_at TEXT
)
"""

# FTS index name used by DuckDB's fts extension.
_FTS_INDEX = "snippets_fts"


class SnippetDB:
    """Sentence-level snippet store backed by DuckDB.

    Stores short text extracts (1-3 sentences) from ingested documents to
    enable fast keyword-based factual lookup without LLM generation.

    Retrieval uses DuckDB's built-in FTS extension (BM25).  If the extension
    is unavailable (e.g. older DuckDB builds) the implementation falls back to
    a case-insensitive ILIKE scan.

    One DuckDB file is shared with MetadataDB — each uses a separate table.
    """

    def __init__(self, db_path: str) -> None:
        import duckdb  # lazy import

        self._db_path = db_path
        self._conn = duckdb.connect(db_path)
        self._fts_ready = False
        self._ensure_table()

    # ------------------------------------------------------------------
    # Schema + FTS
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        self._conn.execute(_DDL)

    def _ensure_fts(self) -> bool:
        """Install and load the FTS extension; create index if needed.

        Returns True when FTS is available, False on failure (triggers ILIKE fallback).
        """
        if self._fts_ready:
            return True
        try:
            self._conn.execute("INSTALL fts")
            self._conn.execute("LOAD fts")
            self._conn.execute(
                "PRAGMA create_fts_index('snippets', 'id', 'text', overwrite=1)"
            )
            self._fts_ready = True
            log.debug("snippet_db.fts.ready")
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("snippet_db.fts.unavailable", error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert_snippets(self, snippets: list[dict[str, Any]]) -> None:
        """Bulk-insert snippet records.

        Each dict must contain: ``id``, ``doc_id``, ``text``.
        Optional keys: ``title``, ``topic``, ``position``, ``source``.
        """
        if not snippets:
            return
        now = datetime.now(tz=timezone.utc).isoformat()
        rows = [
            (
                s["id"],
                s["doc_id"],
                s["text"],
                s.get("title", ""),
                s.get("topic", ""),
                s.get("position", 0),
                s.get("source", ""),
                now,
            )
            for s in snippets
        ]
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO snippets
                (id, doc_id, text, title, topic, position, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        # Rebuild FTS index to include new rows
        self._fts_ready = False
        log.info("snippet_db.insert", count=len(snippets))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search_snippets(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return up to *k* snippets matching *query*.

        Uses DuckDB FTS (BM25) when available, falls back to ILIKE scan.
        Each result dict includes an extra ``score`` key.
        """
        if self._ensure_fts():
            return self._fts_search(query, k)
        return self._ilike_search(query, k)

    def _fts_search(self, query: str, k: int) -> list[dict[str, Any]]:
        try:
            rows = self._conn.execute(
                f"""
                SELECT s.id, s.doc_id, s.text, s.title, s.topic,
                       s.position, s.source, s.created_at,
                       fts.score
                FROM snippets s
                JOIN (
                    SELECT id, score
                    FROM fts_main_snippets.match_bm25(?, fields := 'text')
                ) fts ON s.id = fts.id
                ORDER BY fts.score DESC
                LIMIT ?
                """,
                [query, k],
            ).fetchall()
            return self._rows_to_dicts(rows, with_score=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("snippet_db.fts_search.failed", error=str(exc))
            return self._ilike_search(query, k)

    def _ilike_search(self, query: str, k: int) -> list[dict[str, Any]]:
        # Build a word-boundary ILIKE condition for each query token
        tokens = [t.strip() for t in query.split() if t.strip()]
        if not tokens:
            return []

        conditions = " AND ".join(["text ILIKE ?"] * len(tokens))
        params = [f"%{t}%" for t in tokens] + [k]
        rows = self._conn.execute(
            f"""
            SELECT id, doc_id, text, title, topic, position, source, created_at
            FROM snippets
            WHERE {conditions}
            ORDER BY length(text)
            LIMIT ?
            """,
            params,
        ).fetchall()
        dicts = self._rows_to_dicts(rows, with_score=False)
        for d in dicts:
            d["score"] = 1.0
        return dicts

    def get_snippets_by_doc(self, doc_id: str) -> list[dict[str, Any]]:
        """Return all snippets for a given *doc_id*, ordered by position."""
        rows = self._conn.execute(
            """
            SELECT id, doc_id, text, title, topic, position, source, created_at
            FROM snippets
            WHERE doc_id = ?
            ORDER BY position
            """,
            [doc_id],
        ).fetchall()
        dicts = self._rows_to_dicts(rows, with_score=False)
        for d in dicts:
            d["score"] = 1.0
        return dicts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _COLS = ["id", "doc_id", "text", "title", "topic", "position", "source", "created_at"]

    def _rows_to_dicts(
        self, rows: list[tuple[Any, ...]], *, with_score: bool
    ) -> list[dict[str, Any]]:
        results = []
        for row in rows:
            d = dict(zip(self._COLS, row[:8], strict=True))
            if with_score:
                d["score"] = float(row[8])
            results.append(d)
        return results

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SnippetDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
