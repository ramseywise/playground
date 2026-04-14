"""Human-in-the-loop grader — file-based review queue.

Writes pending review tasks to a JSONL file.  A human reviewer edits
the file to add verdicts.  ``grade()`` checks for a completed review
and returns it, or raises ``PendingReviewError`` if not yet reviewed.

Review entries carry structured tags for failure taxonomy and trace IDs
for linking back to observability traces (OTel / Langfuse).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from eval.models import EvalTask, GraderResult

log = structlog.get_logger(__name__)

# Structured tag vocabulary for CS agent reviews.
# Reviewers fill in zero or more of these in the ``tags`` list.
REVIEW_TAGS: frozenset[str] = frozenset({
    "hallucination",
    "retrieval_relevancy",
    "tone",
    "escalation_failure",
    "context_missing",
})


def validate_tags(tags: list[str]) -> list[str]:
    """Return only recognised review tags, logging a warning for unknowns."""
    valid = [t for t in tags if t in REVIEW_TAGS]
    unknown = [t for t in tags if t not in REVIEW_TAGS]
    if unknown:
        log.warning("human_grader.unknown_tags", unknown=unknown, valid=valid)
    return valid


class PendingReviewError(Exception):
    """Raised when a human review has not been completed yet."""


class HumanGrader:
    """File-based human evaluation grader.

    Workflow:
        1. ``submit(task)`` writes the task to the review queue.
        2. A human edits the review file to add ``is_correct``, ``score``,
           ``reasoning``, and ``tags``.
        3. ``grade(task)`` reads the completed review and returns a ``GraderResult``.

    The review file is JSONL: one object per line, keyed by task ID.
    """

    grader_type: str = "human"

    def __init__(self, review_dir: Path) -> None:
        self._review_dir = review_dir
        self._review_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _queue_path(self) -> Path:
        return self._review_dir / "pending.jsonl"

    @property
    def _completed_path(self) -> Path:
        return self._review_dir / "completed.jsonl"

    def submit(self, task: EvalTask) -> None:
        """Add a task to the pending review queue."""
        entry = {
            "task_id": task.id,
            "query": task.query,
            "expected_answer": task.expected_answer,
            "context": task.context,
            "response": task.metadata.get("response", ""),
            "is_correct": None,
            "score": None,
            "reasoning": "",
            "trace_id": task.metadata.get("trace_id", ""),
            "confidence_score": task.metadata.get("confidence_score", None),
            "tags": [],
        }
        with self._queue_path.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    async def grade(self, task: EvalTask) -> GraderResult:
        """Return the human verdict for this task, or raise PendingReviewError."""
        review = self._find_completed_review(task.id)
        if review is None:
            raise PendingReviewError(f"No completed review for task {task.id}")
        raw_tags: list[str] = review.get("tags", [])
        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            is_correct=bool(review.get("is_correct", False)),
            score=float(review.get("score", 0.0)),
            reasoning=str(review.get("reasoning", "")),
            details={
                "tags": validate_tags(raw_tags) if raw_tags else [],
                "trace_id": review.get("trace_id", ""),
                "confidence_score": review.get("confidence_score"),
            },
        )

    def _find_completed_review(self, task_id: str) -> dict[str, Any] | None:
        """Scan completed reviews for a matching task ID."""
        if not self._completed_path.exists():
            return None
        with self._completed_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("task_id") == task_id and entry.get("is_correct") is not None:
                    return entry
        return None
