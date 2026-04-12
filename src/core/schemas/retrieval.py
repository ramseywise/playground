"""Canonical retrieval result type shared across storage and librarian.

Dependency rule:
    storage → core ← librarian     (correct)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from core.schemas.chunks import Chunk


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    source: Literal["vector", "bm25", "hybrid", "keyword"]
