"""Core evaluation data models.

Generic types for tasks, grader results, and evaluation reports.
Agent-specific models (e.g. librarian's GoldenSample) stay in their
own packages and use conversion helpers to bridge.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from core.config.settings import BaseSettings as _BaseSettings

_settings = _BaseSettings()


# ---------------------------------------------------------------------------
# Task — a single evaluation item
# ---------------------------------------------------------------------------


class EvalTask(BaseModel):
    """Generic QA evaluation task with metadata.

    Agent-specific task types (e.g. GoldenSample) are separate models;
    conversion helpers in ``eval.tasks.extract`` bridge them.
    """

    id: str
    query: str
    expected_answer: str = ""
    context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    category: str = ""
    difficulty: str = "medium"  # easy | medium | hard
    tags: list[str] = Field(default_factory=list)
    validation_level: str = "silver"  # gold | silver | bronze | synthetic


# ---------------------------------------------------------------------------
# Grader result — output from a single grader on a single task
# ---------------------------------------------------------------------------


class GraderResult(BaseModel):
    """Standardised output from any evaluation grader."""

    task_id: str
    grader_type: str
    is_correct: bool
    score: float  # 0.0–1.0
    reasoning: str = ""
    dimensions: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Run configuration — snapshot for reproducibility
# ---------------------------------------------------------------------------


class EvalRunConfig(BaseModel):
    """Configuration snapshot for a single evaluation run.

    Logged alongside metrics so results are reproducible and comparable
    across prompt versions, corpus versions, and retrieval settings.
    """

    run_name: str = ""
    prompt_version: str = "v0.1.0"
    model_id: str = Field(default_factory=lambda: _settings.model_haiku)
    eval_dataset: str = ""
    corpus_version: str = ""
    top_k: int = 5
    notes: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    extra: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return self.model_dump(exclude={"timestamp"}) | {
            "timestamp": self.timestamp.isoformat()
        }


# ---------------------------------------------------------------------------
# Report — aggregate results
# ---------------------------------------------------------------------------


class CategoryBreakdown(BaseModel):
    """Pass rate and score breakdown for a single category slice."""

    category: str
    n_tasks: int
    pass_rate: float
    avg_score: float


class EvalReport(BaseModel):
    """Aggregate evaluation report produced by EvalRunner."""

    run_id: str
    config: EvalRunConfig
    results: list[GraderResult]
    pass_rate: float = 0.0
    avg_score: float = 0.0
    n_tasks: int = 0
    n_passed: int = 0
    by_category: list[CategoryBreakdown] = Field(default_factory=list)
    by_difficulty: list[CategoryBreakdown] = Field(default_factory=list)
    by_grader: list[CategoryBreakdown] = Field(default_factory=list)
    failure_clusters: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Experiment / dashboard shared types
# ---------------------------------------------------------------------------


class QueryResult(BaseModel):
    """Per-query result for a variant experiment.

    Shared by ``experiment.py`` and ``dashboard_data.py`` to avoid
    parallel model hierarchies.
    """

    query_id: str
    query: str
    hit: bool
    reciprocal_rank: float
    retrieved_urls: list[str] = Field(default_factory=list)
    expected_url: str = ""
    latency_ms: float = 0.0
    trace_id: str = ""
    answer: str = ""


class FailureClusterSummary(BaseModel):
    """Failure cluster summary for experiment reports and dashboards."""

    failure_type: str
    count: int
    common_patterns: list[str] = Field(default_factory=list)


class ExperimentResult(BaseModel):
    """Aggregate result of running a variant experiment.

    Shared by ``experiment.py`` and ``dashboard_data.py``.
    """

    variant_name: str
    dataset_name: str = ""
    run_name: str = ""
    hit_rate: float = 0.0
    mrr: float = 0.0
    n_queries: int = 0
    n_hits: int = 0
    avg_latency_ms: float = 0.0
    query_results: list[QueryResult] = Field(default_factory=list)
    failure_clusters: list[FailureClusterSummary] = Field(default_factory=list)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant_name,
            "run_name": self.run_name,
            "hit_rate": round(self.hit_rate, 3),
            "mrr": round(self.mrr, 3),
            "n_queries": self.n_queries,
            "n_hits": self.n_hits,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "failure_types": [
                f"{c.failure_type}\u00d7{c.count}" for c in self.failure_clusters
            ],
        }
