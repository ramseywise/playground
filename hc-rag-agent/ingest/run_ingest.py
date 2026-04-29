"""Offline ingest CLI — build the three-store data layer from raw JSONL sources.

Usage:
    uv run ingest-clara                          # uses .env defaults
    uv run ingest-clara --source data/raw/clara_raw --vectordb data/stores/vectordb/clara.duckdb
    uv run ingest-clara --no-clear               # incremental (append only)

Writes to:
    vectordb  → --vectordb path  (DuckDB: rag_chunks + embedding vectors)
    metadb    → --metadb path    (DuckDB: ingest_documents + ingest_snippets + ingest_runs)
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


def _record_ingest_run(
    metadb_path: Path,
    *,
    corpus: str,
    source_dir: Path,
    vectordb_path: Path,
    embedding_model: str,
    chunk_strategy: str,
    chunk_max_tokens: int,
    chunk_overlap_tokens: int,
    n_docs: int,
    n_chunks: int,
    clear: bool,
) -> None:
    """Write an ingest_runs row so future re-indexes are fully traceable."""
    import duckdb

    metadb_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(metadb_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS ingest_runs (
            run_id        VARCHAR PRIMARY KEY,
            corpus        VARCHAR NOT NULL,
            source_dir    VARCHAR NOT NULL,
            vectordb_path VARCHAR NOT NULL,
            embedding_model VARCHAR NOT NULL,
            chunk_strategy VARCHAR NOT NULL,
            chunk_max_tokens INTEGER,
            chunk_overlap_tokens INTEGER,
            n_docs        INTEGER,
            n_chunks      INTEGER,
            cleared       BOOLEAN,
            ran_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    run_id = f"{corpus}_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    con.execute(
        """
        INSERT INTO ingest_runs
            (run_id, corpus, source_dir, vectordb_path, embedding_model,
             chunk_strategy, chunk_max_tokens, chunk_overlap_tokens,
             n_docs, n_chunks, cleared)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            corpus,
            str(source_dir),
            str(vectordb_path),
            embedding_model,
            chunk_strategy,
            chunk_max_tokens,
            chunk_overlap_tokens,
            n_docs,
            n_chunks,
            clear,
        ],
    )
    con.close()
    log.info("ingest.run_recorded", run_id=run_id, metadb=str(metadb_path))


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv()

    from core.observability import configure_runtime
    configure_runtime()

    parser = argparse.ArgumentParser(description="Ingest JSONL corpus into three-store layout")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.getenv("RAW_DATA_DIR") or "../data/raw/clara_raw"),
        help="Directory of *.jsonl files to ingest",
    )
    parser.add_argument(
        "--vectordb",
        type=Path,
        default=Path(os.getenv("VECTORDB_PATH") or "../data/stores/vectordb/clara.duckdb"),
        help="Destination DuckDB file for vectors + rag_chunks",
    )
    parser.add_argument(
        "--metadb",
        type=Path,
        default=Path(os.getenv("METADB_PATH") or "../data/stores/metadb/clara_meta.db"),
        help="Destination DuckDB file for ingest_documents + ingest_snippets + ingest_runs",
    )
    parser.add_argument(
        "--corpus",
        default="clara",
        help="Corpus name recorded in ingest_runs (e.g. clara, billy)",
    )
    parser.add_argument(
        "--no-clear",
        dest="clear",
        action="store_false",
        default=True,
        help="Incremental mode: append without dropping existing rows",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size",
    )
    args = parser.parse_args()

    from core.config import EMBEDDING_MODEL
    chunk_max = int(os.getenv("RAG_CHUNK_MAX_TOKENS") or "512")
    chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP_TOKENS") or "64")

    log.info(
        "ingest.start",
        corpus=args.corpus,
        source=str(args.source),
        vectordb=str(args.vectordb),
        metadb=str(args.metadb),
        embedding_model=EMBEDDING_MODEL,
        clear=args.clear,
    )

    from rag.ingestion.corpus_v2 import ingest_v2_to_duckdb
    n_chunks = ingest_v2_to_duckdb(
        corpus_dir=args.source,
        duckdb_path=args.vectordb,
        clear=args.clear,
        embed_batch_size=args.batch_size,
    )

    # Count docs from the written index
    import duckdb as _ddb
    con = _ddb.connect(str(args.vectordb), read_only=True)
    n_docs = con.execute("SELECT COUNT(DISTINCT doc_id) FROM ingest_documents").fetchone()[0]
    con.close()

    _record_ingest_run(
        args.metadb,
        corpus=args.corpus,
        source_dir=args.source,
        vectordb_path=args.vectordb,
        embedding_model=EMBEDDING_MODEL,
        chunk_strategy="FixedChunker",
        chunk_max_tokens=chunk_max,
        chunk_overlap_tokens=chunk_overlap,
        n_docs=n_docs,
        n_chunks=n_chunks,
        clear=args.clear,
    )

    log.info("ingest.complete", n_docs=n_docs, n_chunks=n_chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
