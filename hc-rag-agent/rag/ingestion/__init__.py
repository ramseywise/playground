"""Offline corpus ingestion entrypoints (e.g. v2 JSONL → DuckDB)."""

from __future__ import annotations

from rag.ingestion.corpus_v2 import (
    clear_ingestion_tables,
    ingest_v2_to_duckdb,
    load_jsonl_corpus,
)

__all__ = [
    "clear_ingestion_tables",
    "ingest_v2_to_duckdb",
    "load_jsonl_corpus",
]
