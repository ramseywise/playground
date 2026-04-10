"""Pydantic schemas for the golden trace pipeline.

Golden traces are pre-computed, high-quality Q&A pairs extracted from the
corpus.  When a user query closely matches a trace question (via BM25), the
grounded answer is returned directly — bypassing the full RAG pipeline for
sub-50 ms responses.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GoldenTrace(BaseModel):
    """A self-contained knowledge fragment extracted from the corpus."""

    id: str
    text: str
    source_url: str = ""
    source_title: str = ""
    language: str = "en"
    trace_type: str = "text"  # "text" | "code" | "config" | "cli"
    tags: list[str] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    """An answer grounded in one or more source traces."""

    text: str
    trace_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class QAPair(BaseModel):
    """A question-answer pair grounded in source traces.

    Used for direct-hit retrieval: when the user's query closely matches
    the question, the grounded answer is returned without chunked retrieval.
    """

    id: str
    question: str
    answer: GroundedAnswer
    category: str = ""
    difficulty: str = "medium"  # "easy" | "medium" | "hard"
    source_trace_ids: list[str] = Field(default_factory=list)
