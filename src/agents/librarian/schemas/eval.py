"""Evaluation data models for the librarian agent.

``EvalRunConfig`` is re-exported from the shared ``eval.models`` package.
``GoldenSample`` and ``RetrievalMetrics`` are librarian-specific.
"""

from __future__ import annotations

from eval.models import EvalRunConfig  # noqa: F401 — re-export

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


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
