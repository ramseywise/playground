"""Re-export RRF utilities from canonical location: core.retrieval.rrf."""

from __future__ import annotations

from core.retrieval.rrf import RRF_K, _chunk_hash, fuse_rankings  # noqa: F401

__all__ = ["RRF_K", "_chunk_hash", "fuse_rankings"]
