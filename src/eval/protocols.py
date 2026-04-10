"""Base evaluation protocols — shared across all agents.

Each agent's eval harness implements these protocols with domain-specific
grading criteria, golden datasets, and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class JudgeResult:
    """Standardised output from any evaluation judge."""

    is_correct: bool
    score: float  # 0.0–1.0
    reasoning: str
    details: dict[str, Any] | None = None


@runtime_checkable
class Grader(Protocol):
    """Evaluates a single (input, output, expected) triple."""

    async def grade(
        self,
        *,
        query: str,
        response: str,
        context: str = "",
        expected: str = "",
    ) -> JudgeResult: ...


@runtime_checkable
class GoldenDataset(Protocol):
    """Loadable set of golden samples for evaluation."""

    def load(self) -> list[dict[str, Any]]: ...

    @property
    def name(self) -> str: ...
