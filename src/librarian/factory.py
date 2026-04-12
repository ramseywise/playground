"""Backward-compatible re-export — canonical location is ``orchestration.factory``."""

from __future__ import annotations

from orchestration.factory import (  # noqa: F401
    create_ingestion_pipeline,
    create_librarian,
    warm_up_embedder,
)

__all__ = ["create_ingestion_pipeline", "create_librarian", "warm_up_embedder"]
