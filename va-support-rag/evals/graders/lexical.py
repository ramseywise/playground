"""Deterministic graders: exact string match and set overlap."""

from __future__ import annotations

import re
from typing import ClassVar

from evals.utils.models import EvalTask, GraderKind, GraderResult


class ExactMatchGrader:
    """Case-insensitive normalised string comparison."""

    grader_type: str = "exact_match"
    grader_kind: ClassVar[GraderKind] = GraderKind.DETERMINISTIC

    def __init__(self, *, normalize: bool = True) -> None:
        self._normalize = normalize

    async def grade(self, task: EvalTask) -> GraderResult:
        expected = task.expected_answer
        actual = task.metadata.get("response", "")
        if self._normalize:
            expected = _normalize(expected)
            actual = _normalize(actual)
        match = expected == actual
        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            is_correct=match,
            score=1.0 if match else 0.0,
            reasoning="exact match" if match else "mismatch",
        )


class SetOverlapGrader:
    """Token-level set overlap: Jaccard similarity and F1."""

    grader_type: str = "set_overlap"
    grader_kind: ClassVar[GraderKind] = GraderKind.DETERMINISTIC

    def __init__(self, *, threshold: float = 0.5) -> None:
        self._threshold = threshold

    async def grade(self, task: EvalTask) -> GraderResult:
        expected_tokens = _tokenize(task.expected_answer)
        actual_tokens = _tokenize(task.metadata.get("response", ""))

        if not expected_tokens and not actual_tokens:
            return GraderResult(
                task_id=task.id,
                grader_type=self.grader_type,
                is_correct=True,
                score=1.0,
                reasoning="both empty",
            )

        intersection = expected_tokens & actual_tokens
        union = expected_tokens | actual_tokens

        jaccard = len(intersection) / len(union) if union else 0.0
        precision = len(intersection) / len(actual_tokens) if actual_tokens else 0.0
        recall = len(intersection) / len(expected_tokens) if expected_tokens else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )

        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            is_correct=f1 >= self._threshold,
            score=f1,
            reasoning=f"F1={f1:.2f}, Jaccard={jaccard:.2f}",
            dimensions={
                "jaccard": jaccard,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        )


def _normalize(text: str) -> str:
    """Lowercase, strip whitespace and punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def _tokenize(text: str) -> set[str]:
    """Split normalised text into a token set."""
    return set(_normalize(text).split())
