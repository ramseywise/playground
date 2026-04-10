"""Evaluation data models for the librarian agent.

``EvalRunConfig`` is re-exported from the shared ``eval.models`` package.
``GoldenSample`` and ``RetrievalMetrics`` are librarian-specific.
"""

from __future__ import annotations

from eval.models import EvalRunConfig  # noqa: F401 — re-export

from pydantic import BaseModel


class GoldenSample(BaseModel):
    query_id: str
    query: str
    expected_doc_url: str
    relevant_chunk_ids: list[str] = []
    category: str = ""
    language: str = "da"
    difficulty: str = "easy"  # easy | medium | hard
    validation_level: str = "silver"  # gold | silver | bronze | synthetic
    source_ticket_id: str = ""


class RetrievalMetrics(BaseModel):
    hit_rate_at_k: float
    mrr: float
    k: int
    n_queries: int
