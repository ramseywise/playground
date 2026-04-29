"""Corpus v2 (JSONL) → DuckDB: vectors + ``ingest_documents`` / ``ingest_snippets``.

Uses **stable chunk ids** ``{stable_doc_id}_{n}`` (e.g. ``help_10570020_0``) so eval
``relevant_chunk_ids`` align with the legacy notebook-style index.

Environment (optional tuning vs hash-mode defaults):

- ``RAG_CHUNK_MAX_TOKENS`` (default 512)
- ``RAG_CHUNK_OVERLAP_TOKENS`` (default 64)
- ``RAG_CHUNK_MIN_TOKENS`` (default 50)

**Re-ingest:** If you change chunking, embedding model, or id scheme, delete the index
or pass ``clear=True`` so vectors and metadata stay consistent. Mixed old/new rows
break evals and dedup checks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.preprocessing.base import ChunkerConfig

log = logging.getLogger(__name__)

# First key wins when building unified ``text`` for :class:`~app.rag.preprocessing.ingestion.IngestionPipeline`.
_TEXT_FIELD_CANDIDATES: tuple[str, ...] = (
    "text",
    "full_text",
    "content",
    "body",
    "markdown",
    "html",
    "raw_html",
    "page_content",
    "article_body",
    "main_content",
    "clean_text",
    "extracted_text",
    "article_html",
)


def _normalize_v2_doc(doc: dict) -> dict:
    """Ensure ``text`` is set from common scraper / JSONL field names."""
    out: dict = dict(doc)
    if (out.get("text") or "").strip():
        out["text"] = str(out["text"]).strip()
        return out
    for key in _TEXT_FIELD_CANDIDATES:
        if key == "text":
            continue
        raw = out.get(key)
        if isinstance(raw, str) and raw.strip():
            out["text"] = raw.strip()
            return out
    # Non-string content (rare)
    for key in _TEXT_FIELD_CANDIDATES:
        raw = out.get(key)
        if raw is not None and not isinstance(raw, str):
            out["text"] = str(raw).strip()
            if out["text"]:
                return out
    return out


def clear_ingestion_tables(db_path: Path) -> None:
    """Remove all rows from ``rag_chunks`` and ingestion sidecar tables if present."""
    import duckdb

    con = duckdb.connect(str(db_path))
    try:
        con.execute("DELETE FROM rag_chunks")
    except Exception as exc:
        log.warning("corpus_v2.clear.rag_chunks failed: %s", exc)
    for table in ("ingest_documents", "ingest_snippets"):
        try:
            con.execute(f"DELETE FROM {table}")
        except Exception as exc:
            log.debug("corpus_v2.clear.skip table=%s err=%s", table, exc)
    con.close()


def load_jsonl_corpus(corpus_dir: Path) -> list[dict]:
    """Load every ``*.jsonl`` under *corpus_dir*; set ``stable_doc_id`` when missing."""
    from rag.preprocessing.chunking.utils import stable_doc_id_from_document

    docs: list[dict] = []
    skipped_empty = 0
    sample_keys: list[str] | None = None
    for path in sorted(corpus_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
                doc = _normalize_v2_doc(doc)
                if not (doc.get("stable_doc_id") or "").strip():
                    doc["stable_doc_id"] = stable_doc_id_from_document(doc)
                if not (doc.get("text") or "").strip():
                    skipped_empty += 1
                    if sample_keys is None:
                        sample_keys = sorted(doc.keys())[:30]
                    continue
                if not (doc.get("source_file") or "").strip():
                    doc["source_file"] = (
                        (doc.get("url") or "").strip()
                        or (doc.get("stable_doc_id") or "").strip()
                        or f"{path.name}:{line_no}"
                    )
                docs.append(doc)
    if skipped_empty:
        log.warning(
            "corpus_v2.skipped_empty_text n=%d sample_keys=%s",
            skipped_empty,
            sample_keys,
        )
    return docs


def _chunker_config_v2() -> ChunkerConfig:
    from rag.preprocessing.base import ChunkerConfig

    return ChunkerConfig(
        max_tokens=int(os.getenv("RAG_CHUNK_MAX_TOKENS", "512")),
        overlap_tokens=int(os.getenv("RAG_CHUNK_OVERLAP_TOKENS", "64")),
        min_tokens=int(os.getenv("RAG_CHUNK_MIN_TOKENS", "50")),
        chunk_id_mode="stable",
    )


def _apply_vector_store_path(db_path: Path) -> None:
    """Point ``VECTOR_STORE_DIR`` at *db_path* parent so pipeline + retriever share one file."""
    os.environ["VECTOR_STORE_BACKEND"] = "duckdb"
    os.environ["VECTOR_STORE_DIR"] = str(db_path.parent.resolve())


async def _ingest_all(
    docs: list[dict],
    *,
    batch_size: int,
) -> int:
    from rag.preprocessing.chunking.strategies import FixedChunker
    from rag.preprocessing.pipeline import build_ingestion_pipeline
    from rag.datastore.factory import reset_vectorstore_for_tests

    reset_vectorstore_for_tests()
    pipe = build_ingestion_pipeline(
        chunker=FixedChunker(config=_chunker_config_v2()),
        batch_size=batch_size,
    )
    results = await pipe.ingest_documents(docs)
    return sum(r.chunk_count for r in results if not r.skipped)


def ingest_v2_to_duckdb(
    corpus_dir: str | Path,
    duckdb_path: str | Path | None = None,
    *,
    clear: bool = True,
    embed_batch_size: int = 32,
) -> int:
    """Ingest all ``*.jsonl`` documents under *corpus_dir*; return total chunks written.

    Parameters
    ----------
    corpus_dir:
        Directory containing one or more ``*.jsonl`` files (one JSON object per line).
        Each object should include at least a text field (``text``, ``full_text``, or
        ``content``) plus ``url`` / ``title`` / ``source`` where available.
    duckdb_path:
        Path to ``rag_index.duckdb``. Defaults to :func:`~app.rag.retrieval.runtime.get_duckdb_index_path`.
    clear:
        If True, delete all rows in ``rag_chunks`` and ingestion tables before loading.
    embed_batch_size:
        Batch size for embedding + vector upsert.
    """
    from rag.datastore.factory import (
        get_duckdb_index_path,
        reset_vectorstore_for_tests,
    )

    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus dir not found: {corpus_dir}")

    db_path = Path(duckdb_path).resolve() if duckdb_path else get_duckdb_index_path()
    _apply_vector_store_path(db_path)

    if clear:
        clear_ingestion_tables(db_path)
        reset_vectorstore_for_tests()

    docs = load_jsonl_corpus(corpus_dir)
    if not docs:
        log.warning("corpus_v2.empty dir=%s", corpus_dir)
        return 0

    total = asyncio.run(_ingest_all(docs, batch_size=embed_batch_size))
    log.info(
        "corpus_v2.done n_docs=%d n_chunks=%d path=%s",
        len(docs),
        total,
        db_path,
    )
    return total


def main() -> int:
    from core.observability import configure_runtime

    configure_runtime()
    p = argparse.ArgumentParser(
        description="Ingest corpus v2 JSONL into DuckDB (stable chunk ids + metadata)."
    )
    p.add_argument(
        "corpus_dir",
        type=Path,
        nargs="?",
        default=Path("data/raptor_scraper/output/v2"),
        help="Directory of *.jsonl (default: data/raptor_scraper/output/v2)",
    )
    p.add_argument(
        "--duckdb",
        type=Path,
        default=None,
        help="Path to rag_index.duckdb (default: VECTOR_STORE_DIR/rag_index.duckdb)",
    )
    p.add_argument(
        "--no-clear",
        action="store_true",
        help="Append without deleting existing rag_chunks / ingest tables",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding upsert batch size",
    )
    args = p.parse_args()
    n = ingest_v2_to_duckdb(
        args.corpus_dir,
        duckdb_path=args.duckdb,
        clear=not args.no_clear,
        embed_batch_size=args.batch_size,
    )
    print(f"Indexed {n} chunks")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
