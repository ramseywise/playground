"""Retrieval result and query schemas."""

from __future__ import annotations

from pydantic import BaseModel

from rag.schemas.chunks import Chunk


class RetrievalResult(BaseModel):
    """Raw output from a single retriever search call."""

    chunk: Chunk
    score: float
    source: str = ""


__all__ = ["RetrievalResult"]
