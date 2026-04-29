"""Shared eval models across all VA services."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class EvalTask(BaseModel):
    """Single evaluation task — one Clara ticket or test case."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    expected_intent: str | None = None
    expected_answer: str | None = None
    ces_rating: int | None = None
    test_type: str | None = None
    source: str = "clara_raw"
    language: str = "de"
    source_category: str | None = None
    escalation_signal: bool = False
    category: str = "general"
    difficulty: str = "medium"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ServiceResponse(BaseModel):
    """Normalized response from any VA service."""

    service: str  # "va-google-adk", "va-langgraph", "va-support-rag"
    task_id: str
    raw_response: dict[str, Any]
    message: str
    suggestions: list[str] = Field(default_factory=list)
    nav_buttons: list[dict] = Field(default_factory=list)
    classified_intent: str | None = None
    latency_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraderResult(BaseModel):
    """Result of grading a single task with a single grader."""

    task_id: str
    grader_type: str
    service: str
    is_correct: bool
    score: float
    reasoning: str
    dimensions: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class EvalReport(BaseModel):
    """Aggregated eval results across all tasks and services."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    run_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    n_tasks: int
    results: list[GraderResult]
    by_service: dict[str, dict[str, Any]] = Field(default_factory=dict)
    by_grader: dict[str, dict[str, Any]] = Field(default_factory=dict)
