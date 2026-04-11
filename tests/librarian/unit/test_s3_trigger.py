"""Tests for the S3 event-triggered Lambda handler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@dataclass
class _FakeResult:
    doc_id: str = "abc123"
    chunk_count: int = 5
    snippet_count: int = 3
    skipped: bool = False


def _make_s3_event(*keys: str, bucket: str = "my-bucket") -> dict[str, Any]:
    """Build a minimal S3 ObjectCreated event with the given keys."""
    return {
        "Records": [
            {"s3": {"bucket": {"name": bucket}, "object": {"key": k}}}
            for k in keys
        ]
    }


class TestS3TriggerHandler:
    def test_single_record(self) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.ingest_s3_object = AsyncMock(return_value=_FakeResult())

        with patch(
            "interfaces.api.s3_trigger._get_pipeline",
            return_value=mock_pipeline,
        ):
            from interfaces.api.s3_trigger import handler

            result = handler(_make_s3_event("raw/doc.md"), context=None)

        assert len(result["ingested"]) == 1
        assert result["ingested"][0]["key"] == "raw/doc.md"
        assert result["ingested"][0]["doc_id"] == "abc123"
        mock_pipeline.ingest_s3_object.assert_called_once()

    def test_multiple_records(self) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.ingest_s3_object = AsyncMock(return_value=_FakeResult())

        with patch(
            "interfaces.api.s3_trigger._get_pipeline",
            return_value=mock_pipeline,
        ):
            from interfaces.api.s3_trigger import handler

            result = handler(
                _make_s3_event("raw/a.md", "raw/b.md", "raw/c.md"), context=None
            )

        assert len(result["ingested"]) == 3
        assert mock_pipeline.ingest_s3_object.call_count == 3

    def test_empty_records(self) -> None:
        mock_pipeline = MagicMock()

        with patch(
            "interfaces.api.s3_trigger._get_pipeline",
            return_value=mock_pipeline,
        ):
            from interfaces.api.s3_trigger import handler

            result = handler({"Records": []}, context=None)

        assert result["ingested"] == []
        mock_pipeline.ingest_s3_object.assert_not_called()

    def test_skipped_document(self) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.ingest_s3_object = AsyncMock(
            return_value=_FakeResult(skipped=True, doc_id="", chunk_count=0)
        )

        with patch(
            "interfaces.api.s3_trigger._get_pipeline",
            return_value=mock_pipeline,
        ):
            from interfaces.api.s3_trigger import handler

            result = handler(_make_s3_event("raw/dup.md"), context=None)

        assert result["ingested"][0]["skipped"] is True
