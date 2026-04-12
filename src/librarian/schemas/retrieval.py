"""Re-export from canonical location: schemas.queries.

All query/retrieval models now live in ``agents.librarian.schemas.queries``.
This module re-exports for backward compatibility.
"""

from __future__ import annotations

from librarian.schemas.queries import (  # noqa: F401
    Intent,
    QueryPlan,
    RetrievalResult,
)

__all__ = ["Intent", "QueryPlan", "RetrievalResult"]
