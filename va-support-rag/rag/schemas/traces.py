"""Golden trace and QA pair models for evaluation dataset generation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GoldenTrace(BaseModel):
    """A self-contained passage extracted from a source document."""

    id: str
    text: str
    source_url: str = ""
    source_title: str = ""
    language: str = "en"
    trace_type: str = "text"
    tags: list[str] = Field(default_factory=list)


class QAPair(BaseModel):
    """A generated question-answer pair grounded in a GoldenTrace."""

    id: str = ""
    question: str = ""
    answer: str = ""
    trace_id: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


__all__ = ["GoldenTrace", "QAPair"]
