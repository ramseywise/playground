"""Ragas library wrapper — maps ragas metrics to GraderResult.

Wraps ``ragas`` evaluation metrics (faithfulness, answer_relevancy,
context_precision, context_recall) behind the ``Grader`` protocol.

Requires the ``ragas`` optional dependency.
"""

from __future__ import annotations

from typing import Any

from eval.models import EvalTask, GraderResult

METRIC_REGISTRY: dict[str, str] = {
    "faithfulness": "ragas.metrics.faithfulness",
    "answer_relevancy": "ragas.metrics.answer_relevancy",
    "context_precision": "ragas.metrics.context_precision",
    "context_recall": "ragas.metrics.context_recall",
    "answer_similarity": "ragas.metrics.answer_similarity",
    "answer_correctness": "ragas.metrics.answer_correctness",
}


class RagasGrader:
    """Evaluate tasks using ragas library metrics.

    Args:
        metrics: List of metric names from ``METRIC_REGISTRY``.
                 Defaults to faithfulness + answer_relevancy.
        threshold: Minimum score to pass (0.0-1.0).
    """

    grader_type: str = "ragas"

    def __init__(
        self,
        metrics: list[str] | None = None,
        *,
        threshold: float = 0.5,
    ) -> None:
        self._metric_names = metrics or ["faithfulness", "answer_relevancy"]
        self._threshold = threshold
        self._metrics: list[Any] = []
        self._initialised = False

    def _ensure_initialised(self) -> None:
        """Lazy-import ragas and resolve metric objects."""
        if self._initialised:
            return
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            answer_similarity,
            context_precision,
            context_recall,
            faithfulness,
        )

        name_to_metric: dict[str, Any] = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_similarity": answer_similarity,
            "answer_correctness": answer_correctness,
        }
        for name in self._metric_names:
            metric = name_to_metric.get(name)
            if metric is None:
                msg = f"Unknown ragas metric: {name!r}. Available: {sorted(name_to_metric)}"
                raise ValueError(msg)
            self._metrics.append((name, metric))
        self._initialised = True

    async def grade(self, task: EvalTask) -> GraderResult:
        """Run ragas metrics on a single task."""
        self._ensure_initialised()
        from eval.datasets import Dataset
        from ragas import evaluate

        response = task.metadata.get("response", "")
        contexts = task.metadata.get("contexts", [])
        if not contexts and task.context:
            contexts = [task.context]

        data = {
            "question": [task.query],
            "answer": [response],
            "contexts": [contexts],
            "ground_truth": [task.expected_answer],
        }
        dataset = Dataset.from_dict(data)

        metric_objs = [m for _, m in self._metrics]
        result = evaluate(dataset, metrics=metric_objs)

        dimensions: dict[str, float] = {}
        for name, _ in self._metrics:
            score = result.get(name)
            if score is not None:
                dimensions[name] = float(score)

        avg_score = sum(dimensions.values()) / len(dimensions) if dimensions else 0.0
        is_correct = avg_score >= self._threshold

        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            is_correct=is_correct,
            score=avg_score,
            reasoning=f"ragas scores: {dimensions}",
            dimensions=dimensions,
        )
