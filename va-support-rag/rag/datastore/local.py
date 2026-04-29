"""Local vector stores — in-process dict, Chroma persistent, or DuckDB file.

All implement :class:`~app.rag.retrieval.protocols.Retriever` plus
``similarity_search_with_score(query, k)`` for the LangGraph retriever node.

Install optional ``rag`` extras for Chroma/DuckDB (``chromadb``, ``duckdb``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from rag.embedding import get_embeddings
from rag.schemas.chunks import Chunk, ChunkMetadata
from rag.schemas.retrieval import RetrievalResult

log = logging.getLogger(__name__)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _chunk_to_lc_metadata(chunk: Chunk) -> dict[str, str]:
    m = chunk.metadata
    return {
        "id": chunk.id,
        "url": m.url or "",
        "title": m.title or "",
        "doc_id": m.doc_id or "",
        "topic": m.topic or "",
        "section": m.section or "",
        "source": m.source_id or "",
    }


def _metadata_to_chunk(meta: dict[str, Any], text: str, chunk_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            url=str(meta.get("url") or ""),
            title=str(meta.get("title") or ""),
            doc_id=str(meta.get("doc_id") or ""),
            topic=meta.get("topic") or None,
            section=meta.get("section") or None,
            source_id=meta.get("source") or None,
        ),
    )


class DictVectorIndex:
    """Pure-Python memory index — embeddings stored per chunk id."""

    def __init__(self) -> None:
        self._embeddings = get_embeddings()
        self._rows: dict[str, tuple[list[float], str, dict[str, str]]] = {}

    def doc_count(self) -> int:
        return len(self._rows)

    def upsert_flat(
        self,
        items: list[tuple[str, str, dict[str, str], list[float]]],
    ) -> None:
        """Insert precomputed embeddings (one row per text). Used by bootstrap."""
        for eid, text, meta, vec in items:
            self._rows[eid] = (vec, text, meta)

    def clear(self) -> None:
        self._rows.clear()

    def load_from_json_file(self, path: Path) -> int:
        """Load id/text/embedding records written by the legacy JSON export."""
        if not path.is_file():
            return 0
        raw = json.loads(path.read_text(encoding="utf-8"))
        n = 0
        for rec in raw:
            eid = str(rec["id"])
            emb = rec["embedding"]
            text = str(rec["content"])
            fn = str(rec.get("filename", ""))
            meta = {"filename": fn, "source": fn or "document"}
            if isinstance(emb, list):
                self._rows[eid] = (emb, text, meta)
                n += 1
        log.info("dict_index.loaded_json path=%s n=%s", path, n)
        return n

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        if not self._rows:
            return []
        q = self._embeddings.embed_query(query)
        ranked: list[tuple[float, str, str, dict[str, str]]] = []
        for eid, (vec, text, meta) in self._rows.items():
            ranked.append((_dot(q, vec), eid, text, meta))
        ranked.sort(key=lambda x: x[0], reverse=True)
        out: list[tuple[Document, float]] = []
        for score, eid, text, meta in ranked[:k]:
            doc = Document(page_content=text, metadata={**meta, "id": eid})
            out.append((doc, score))
        return out

    async def upsert(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.id!r} has no embedding")
            self._rows[c.id] = (c.embedding, c.text, _chunk_to_lc_metadata(c))

    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        k: int = 10,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        _ = metadata_filter
        if not self._rows:
            return []
        ranked: list[tuple[float, str, str, dict[str, str], list[float]]] = []
        for eid, (vec, text, meta) in self._rows.items():
            ranked.append((_dot(query_vector, vec), eid, text, meta, vec))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievalResult(
                chunk=_metadata_to_chunk(r[3], r[2], r[1]),
                score=r[0],
                source=str(r[3].get("url") or ""),
            )
            for r in ranked[:k]
        ]


class ChromaVectorIndex:
    """Persistent Chroma collection (cosine distance)."""

    def __init__(
        self, persist_directory: Path, collection_name: str = "rag_chunks"
    ) -> None:
        import chromadb  # type: ignore[import-untyped]
        from chromadb.config import Settings  # type: ignore[import-untyped]

        persist_directory.mkdir(parents=True, exist_ok=True)
        self._embeddings = get_embeddings()
        self._client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def doc_count(self) -> int:
        return int(self._collection.count())

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        n_docs = self._collection.count()
        if n_docs == 0:
            return []
        q = self._embeddings.embed_query(query)
        res = self._collection.query(
            query_embeddings=[q],  # type: ignore[arg-type]
            n_results=min(k, n_docs),
            include=["documents", "distances", "metadatas"],
        )
        if not res["ids"] or not res["ids"][0]:
            return []
        out: list[tuple[Document, float]] = []
        for i, eid in enumerate(res["ids"][0]):
            text = res["documents"][0][i] if res["documents"] else ""
            meta = dict(res["metadatas"][0][i] or {}) if res["metadatas"] else {}  # type: ignore[index]
            dist = res["distances"][0][i] if res["distances"] else 0.0
            score = 1.0 - float(dist)
            doc = Document(page_content=text or "", metadata={**meta, "id": eid})
            out.append((doc, score))
        return out

    def upsert_blocking(self, chunks: list[Chunk]) -> None:
        """Synchronous upsert (bootstrap / threaded callers)."""
        if not chunks:
            return
        ids = [c.id for c in chunks]
        embs = []
        docs = []
        metas = []
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.id!r} has no embedding")
            embs.append(c.embedding)
            docs.append(c.text)
            metas.append(_chunk_to_lc_metadata(c))
        self._collection.upsert(
            ids=ids,
            embeddings=embs,  # type: ignore[arg-type]
            documents=docs,
            metadatas=metas,  # type: ignore[arg-type]
        )

    async def upsert(self, chunks: list[Chunk]) -> None:
        await asyncio.to_thread(self.upsert_blocking, chunks)

    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        k: int = 10,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        _ = query_text
        _ = metadata_filter

        def _run() -> list[RetrievalResult]:
            n = self._collection.count()
            if n == 0:
                return []
            res = self._collection.query(
                query_embeddings=[query_vector],  # type: ignore[arg-type]
                n_results=min(k, n),
                include=["documents", "distances", "metadatas"],
            )
            if not res["ids"] or not res["ids"][0]:
                return []
            out: list[RetrievalResult] = []
            for i, eid in enumerate(res["ids"][0]):
                text = res["documents"][0][i] if res["documents"] else ""
                meta = dict(res["metadatas"][0][i] or {}) if res["metadatas"] else {}  # type: ignore[index]
                dist = res["distances"][0][i] if res["distances"] else 0.0
                score = 1.0 - float(dist)
                ch = _metadata_to_chunk(meta, text or "", str(eid))
                out.append(
                    RetrievalResult(
                        chunk=ch, score=score, source=str(meta.get("url") or "")
                    ),
                )
            return out

        return await asyncio.to_thread(_run)


class DuckDBVectorIndex:
    """Single-table DuckDB file with FLOAT[] embeddings (cosine via dot product).

    Stores chunk text plus metadata (url, title, doc_id, source, topic, section).
    """

    def __init__(self, database_path: Path) -> None:
        import duckdb  # type: ignore[import-untyped]

        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._embeddings = get_embeddings()
        self._path = database_path
        self._con = duckdb.connect(str(database_path))
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id VARCHAR PRIMARY KEY,
                text VARCHAR,
                url VARCHAR,
                title VARCHAR,
                doc_id VARCHAR,
                source VARCHAR,
                topic VARCHAR,
                section VARCHAR,
                embedding FLOAT[]
            )
        """)
        self._migrate_rag_chunks_columns()

    def _migrate_rag_chunks_columns(self) -> None:
        rows = self._con.execute("PRAGMA table_info('rag_chunks')").fetchall()
        names = {str(row[1]) for row in rows}
        for col in ("source", "topic", "section"):
            if col not in names:
                self._con.execute(f"ALTER TABLE rag_chunks ADD COLUMN {col} VARCHAR")

    def doc_count(self) -> int:
        row = self._con.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()
        return int(row[0]) if row else 0

    def clear_blocking(self) -> None:
        self._con.execute("DELETE FROM rag_chunks")

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        q = self._embeddings.embed_query(query)
        rows = self._con.execute(
            """
            SELECT id, text, url, title, doc_id,
                   COALESCE(source, ''), COALESCE(topic, ''), COALESCE(section, ''),
                   embedding
            FROM rag_chunks
            """,
        ).fetchall()
        if not rows:
            return []
        ranked: list[tuple[Any, ...]] = []
        for row in rows:
            eid, text, url, title, doc_id, source, topic, section, emb = row
            vec = list(emb) if emb is not None else []
            ranked.append(
                (_dot(q, vec), eid, text, url, title, doc_id, source, topic, section),
            )
        ranked.sort(key=lambda x: x[0], reverse=True)
        out: list[tuple[Document, float]] = []
        for (
            score,
            eid,
            text,
            url,
            title,
            doc_id,
            source,
            topic,
            section,
        ) in ranked[:k]:
            meta = {
                "id": eid,
                "url": url or "",
                "title": title or "",
                "doc_id": doc_id or "",
                "source": source or "",
                "topic": topic or "",
                "section": section or "",
            }
            doc = Document(page_content=text or "", metadata=meta)
            out.append((doc, score))
        return out

    def upsert_blocking(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.id!r} has no embedding")
            m = c.metadata
            self._con.execute(
                """
                INSERT OR REPLACE INTO rag_chunks
                    (id, text, url, title, doc_id, source, topic, section, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    c.id,
                    c.text,
                    m.url or "",
                    m.title or "",
                    m.doc_id or "",
                    m.source_id or "",
                    m.topic or "",
                    m.section or "",
                    c.embedding,
                ],
            )

    async def upsert(self, chunks: list[Chunk]) -> None:
        await asyncio.to_thread(self.upsert_blocking, chunks)

    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        k: int = 10,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        _ = query_text
        _ = metadata_filter

        rows = self._con.execute(
            """
            SELECT id, text, url, title, doc_id,
                   COALESCE(source, ''), COALESCE(topic, ''), COALESCE(section, ''),
                   embedding
            FROM rag_chunks
            """,
        ).fetchall()
        if not rows:
            return []
        ranked: list[tuple[float, str, str, str, str, str, str, str, str]] = []
        for row in rows:
            eid, text, url, title, doc_id, source, topic, section, emb = row
            vec = list(emb) if emb is not None else []
            ranked.append(
                (
                    _dot(query_vector, vec),
                    eid,
                    text,
                    url or "",
                    title or "",
                    doc_id or "",
                    source or "",
                    topic or "",
                    section or "",
                ),
            )
        ranked.sort(key=lambda x: x[0], reverse=True)
        out: list[RetrievalResult] = []
        for (
            score,
            eid,
            text,
            url,
            title,
            doc_id,
            source,
            topic,
            section,
        ) in ranked[:k]:
            meta = {
                "url": url,
                "title": title,
                "doc_id": doc_id,
                "id": eid,
                "source": source,
                "topic": topic,
                "section": section,
            }
            ch = _metadata_to_chunk(meta, text or "", eid)
            out.append(RetrievalResult(chunk=ch, score=score, source=url or ""))
        return out


__all__ = [
    "ChromaVectorIndex",
    "DictVectorIndex",
    "DuckDBVectorIndex",
]
