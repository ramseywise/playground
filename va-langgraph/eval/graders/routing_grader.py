"""Deterministic routing grader: expected_intent == classified_intent."""

from __future__ import annotations

from ..models import EvalTask, GraderResult


class RoutingGrader:
    """Checks that task.metadata['classified_intent'] matches task.expected_intent."""

    grader_type = "routing"

    async def grade(self, task: EvalTask) -> GraderResult:
        expected = task.expected_intent
        actual = task.metadata.get("classified_intent")

        if expected is None:
            return GraderResult(
                task_id=task.id,
                grader_type=self.grader_type,
                is_correct=True,
                score=1.0,
                reasoning="No expected_intent — skipped.",
            )

        is_correct = actual == expected

        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            reasoning=f"Expected {expected!r}, got {actual!r}.",
            dimensions={"match": float(is_correct)},
            details={"expected_intent": expected, "classified_intent": actual},
        )
