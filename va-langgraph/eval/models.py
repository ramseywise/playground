"""Core data models for the VA LangGraph eval framework."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class EvalTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    query: str

    # Routing tasks
    expected_intent: str | None = None

    # Safety tasks
    expected_blocked: bool | None = None
    contains_pii: bool = False
    pii_tokens: list[str] = Field(default_factory=list)

    category: str = "general"
    difficulty: str = "medium"
    tags: list[str] = Field(default_factory=list)

    # Populated by the test harness before grading
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraderResult(BaseModel):
    task_id: str
    grader_type: str
    is_correct: bool
    score: float
    reasoning: str
    dimensions: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class CategoryBreakdown(BaseModel):
    category: str
    n_tasks: int
    n_passed: int
    pass_rate: float
    avg_score: float


class EvalRunConfig(BaseModel):
    run_name: str
    model_id: str = "gemini-2.5-flash"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)


class EvalReport(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    config: EvalRunConfig
    results: list[GraderResult]
    pass_rate: float
    avg_score: float
    n_tasks: int
    n_passed: int
    by_category: list[CategoryBreakdown] = Field(default_factory=list)
    by_grader: list[CategoryBreakdown] = Field(default_factory=list)
    failure_details: list[dict[str, Any]] = Field(default_factory=list)
