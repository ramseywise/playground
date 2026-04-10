"""Shared evaluation framework — base protocols, graders, pipelines, and runner.

Agent-specific eval (e.g. librarian RAG eval) extends these base classes.
Dependency direction: agents.*.eval → eval → core (never reversed).
"""

from eval.models import (
    CategoryBreakdown,
    EvalReport,
    EvalRunConfig,
    EvalTask,
    GraderResult,
)
from eval.protocols import Grader, GoldenDataset
from eval.runner import EvalRunner

__all__ = [
    "CategoryBreakdown",
    "EvalReport",
    "EvalRunConfig",
    "EvalRunner",
    "EvalTask",
    "GoldenDataset",
    "Grader",
    "GraderResult",
]
