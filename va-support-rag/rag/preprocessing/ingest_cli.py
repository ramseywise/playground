"""CLI: chunk + embed + upsert Markdown or FAQ CSV into the vector backend + DuckDB sidecar.

Environment:
  ``VECTOR_STORE_BACKEND`` — ``duckdb`` | ``opensearch`` | ``memory`` (see ``app.core.config``)
  OpenSearch: ``OPENSEARCH_HOSTS``, ``OPENSEARCH_INDEX``, …

Examples::

    VECTOR_STORE_BACKEND=duckdb uv run --extra rag python -m app.rag.preprocessing.ingest_cli --directory data/document
    VECTOR_STORE_BACKEND=duckdb uv run --extra rag python -m app.rag.preprocessing.ingest_cli --csv data/scraped_faq.csv
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from core.observability import configure_runtime

log = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest Markdown or FAQ CSV into RAG vector index + metadata DB."
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--directory",
        "-d",
        type=Path,
        default=None,
        help="Directory of Markdown files (default: data/document if no --csv)",
    )
    src.add_argument(
        "--csv",
        "-c",
        type=Path,
        default=None,
        help="Single scraped FAQ CSV (url + question/answer columns; see faq_csv module)",
    )
    p.add_argument(
        "--glob",
        "-g",
        default="*.md",
        help="Glob under directory (default: *.md)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_runtime()
    args = _parse_args(argv)
    directory = args.directory if args.directory is not None else Path("data/document")

    from rag.preprocessing.pipeline import (
        build_ingestion_pipeline,
        load_faq_csv_documents,
    )

    if args.csv is not None:
        if not args.csv.is_file():
            log.error("ingest_cli: not a file: %s", args.csv)
            return 1
    elif not directory.is_dir():
        log.error("ingest_cli: not a directory: %s", directory)
        return 1

    async def _run() -> None:
        if args.csv is not None:
            from rag.preprocessing.base import ChunkerConfig
            from rag.preprocessing.chunking.strategies import FixedChunker

            pipe = build_ingestion_pipeline(
                chunker=FixedChunker(
                    config=ChunkerConfig(chunk_id_mode="stable"),
                ),
            )
            docs = load_faq_csv_documents(args.csv)
            results = await pipe.ingest_documents(docs)
        else:
            pipe = build_ingestion_pipeline()
            results = await pipe.ingest_directory(directory, glob_pattern=args.glob)
        ok = sum(1 for r in results if not r.skipped)
        skipped = len(results) - ok
        log.info(
            "ingest_cli.done items=%d indexed=%d skipped=%d",
            len(results),
            ok,
            skipped,
        )

    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
