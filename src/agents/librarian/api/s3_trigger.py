"""Lambda handler for S3 event-triggered ingestion.

Receives S3 ObjectCreated events on the ``raw/`` prefix.
Creates an ``IngestionPipeline`` and ingests each uploaded document.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)

_pipeline: Any = None


def _get_pipeline() -> Any:
    """Lazy-initialize the ingestion pipeline (persists across warm invocations)."""
    global _pipeline  # noqa: PLW0603
    if _pipeline is None:
        from agents.librarian.factory import create_ingestion_pipeline

        _pipeline = create_ingestion_pipeline()
        log.info("s3_trigger.pipeline.init")
    return _pipeline


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entry point for S3 event notifications."""
    pipeline = _get_pipeline()
    results: list[dict[str, Any]] = []

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        log.info("s3_trigger.ingest", bucket=bucket, key=key)

        result = asyncio.run(
            pipeline.ingest_s3_object(bucket=bucket, key=key)
        )
        results.append({
            "key": key,
            "doc_id": result.doc_id,
            "chunk_count": result.chunk_count,
            "skipped": result.skipped,
        })

    log.info("s3_trigger.done", record_count=len(results))
    return {"ingested": results}
