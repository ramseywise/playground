"""Pydantic schemas for the snippet DB pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Snippet(BaseModel):
    """A self-contained code or text snippet extracted from the corpus."""

    id: str
    text: str
    source_url: str = ""
    source_title: str = ""
    language: str = "en"
    snippet_type: str = "text"  # "text" | "code" | "config" | "cli"
    tags: list[str] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    """An answer grounded in one or more source snippets."""

    text: str
    snippet_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class QAPair(BaseModel):
    """A question-answer pair grounded in source snippets.

    Used for direct-hit retrieval: when the user's query closely matches
    the question, the grounded answer is returned without chunked retrieval.
    """

    id: str
    question: str
    answer: GroundedAnswer
    category: str = ""
    difficulty: str = "medium"  # "easy" | "medium" | "hard"
    source_snippet_ids: list[str] = Field(default_factory=list)
