"""Base evaluation protocols — shared across all agents.

Each agent's eval harness implements these protocols with domain-specific
grading criteria, golden datasets, and metrics.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agents.librarian.eval.models import EvalTask, GraderResult


@runtime_checkable
class Grader(Protocol):
    """Evaluates a single task and returns a standardised result."""

    @property
    def grader_type(self) -> str: ...

    async def grade(self, task: EvalTask) -> GraderResult: ...


@runtime_checkable
class GoldenDataset(Protocol):
    """Loadable set of golden samples for evaluation."""

    def load(self) -> list[dict[str, Any]]: ...

    @property
    def name(self) -> str: ...
