"""Multiple-choice question grader."""

from __future__ import annotations

import re

from agents.librarian.eval.models import EvalTask, GraderResult


class MCQGrader:
    """Grades multiple-choice answers by matching option letters.

    Expects ``task.expected_answer`` to be the correct option letter (e.g. "B").
    Reads the model response from ``task.metadata["response"]`` and extracts
    the first letter A-Z found.
    """

    grader_type: str = "mcq"

    async def grade(self, task: EvalTask) -> GraderResult:
        expected = _extract_letter(task.expected_answer)
        response = task.metadata.get("response", "")
        actual = _extract_letter(response)

        if not expected:
            return GraderResult(
                task_id=task.id,
                grader_type=self.grader_type,
                is_correct=False,
                score=0.0,
                reasoning="no expected answer letter found",
            )

        match = expected == actual
        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            is_correct=match,
            score=1.0 if match else 0.0,
            reasoning=f"expected={expected}, got={actual or 'none'}",
            details={"expected_letter": expected, "actual_letter": actual},
        )


_LETTER_RE = re.compile(r"\b([A-Za-z])\b")


def _extract_letter(text: str) -> str:
    """Extract the first standalone letter from text, uppercased."""
    m = _LETTER_RE.search(text.strip())
    return m.group(1).upper() if m else ""
