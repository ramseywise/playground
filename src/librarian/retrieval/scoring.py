"""Re-export scoring primitives from canonical location: core.retrieval.scoring."""

from __future__ import annotations

from core.retrieval.scoring import cosine_similarity, term_overlap  # noqa: F401

__all__ = ["cosine_similarity", "term_overlap"]
