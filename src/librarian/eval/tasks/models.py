"""Librarian-specific evaluation data models.

``GoldenSample`` and ``RetrievalMetrics`` are owned here.
Generic models (``EvalTask``, ``GraderResult``, ``EvalReport``, ``EvalRunConfig``)
live in ``agents.librarian.eval.models``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GoldenSample(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query_id: str
    query: str
    expected_doc_url: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    category: str = ""
    language: str = "en"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    validation_level: Literal["gold", "silver", "bronze", "synthetic"] = "silver"
    source_record_id: str | None = Field(default=None, alias="source_ticket_id")


class RetrievalMetrics(BaseModel):
    hit_rate_at_k: float
    mrr: float
    k: int
    n_queries: int
