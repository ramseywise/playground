"""Structured RAG response model.

Used by the generation layer when the intent is retrieval-based
(lookup, explore, compare).  Conversational and out-of-scope intents
return unstructured text and bypass this model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Citation(BaseModel):
    """A single source citation extracted from the LLM response."""

    url: str
    title: str
    snippet: str = ""


class RAGResponse(BaseModel):
    """Structured answer from the RAG pipeline."""

    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]
    follow_up: str = ""
