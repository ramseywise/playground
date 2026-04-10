"""DeepEval library wrapper — maps deepeval metrics to GraderResult.

Wraps ``deepeval`` evaluation metrics (faithfulness, answer relevancy,
contextual precision/recall) behind the ``Grader`` protocol.

Requires the ``deepeval`` optional dependency.
"""

from __future__ import annotations

from typing import Any

from eval.models import EvalTask, GraderResult

# Available metric names and their deepeval class paths
METRIC_REGISTRY: dict[str, str] = {
    "faithfulness": "deepeval.metrics.FaithfulnessMetric",
    "answer_relevancy": "deepeval.metrics.AnswerRelevancyMetric",
    "contextual_precision": "deepeval.metrics.ContextualPrecisionMetric",
    "contextual_recall": "deepeval.metrics.ContextualRecallMetric",
    "contextual_relevancy": "deepeval.metrics.ContextualRelevancyMetric",
    "hallucination": "deepeval.metrics.HallucinationMetric",
}


class DeepEvalGrader:
    """Evaluate tasks using deepeval library metrics.

    Args:
        metrics: List of metric names from ``METRIC_REGISTRY``.
                 Defaults to faithfulness + answer_relevancy.
        threshold: Minimum score to pass (0.0-1.0).
        model: Model name for deepeval's internal LLM calls.
    """

    grader_type: str = "deepeval"

    def __init__(
        self,
        metrics: list[str] | None = None,
        *,
        threshold: float = 0.5,
        model: str | None = None,
    ) -> None:
        self._metric_names = metrics or ["faithfulness", "answer_relevancy"]
        self._threshold = threshold
        self._model = model
        self._metrics: list[Any] = []
        self._initialised = False

    def _ensure_initialised(self) -> None:
        """Lazy-import deepeval and instantiate metrics."""
        if self._initialised:
            return
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
            HallucinationMetric,
        )

        name_to_cls: dict[str, type] = {
            "faithfulness": FaithfulnessMetric,
            "answer_relevancy": AnswerRelevancyMetric,
            "contextual_precision": ContextualPrecisionMetric,
            "contextual_recall": ContextualRecallMetric,
            "contextual_relevancy": ContextualRelevancyMetric,
            "hallucination": HallucinationMetric,
        }
        for name in self._metric_names:
            cls = name_to_cls.get(name)
            if cls is None:
                msg = f"Unknown deepeval metric: {name!r}. Available: {sorted(name_to_cls)}"
                raise ValueError(msg)
            kwargs: dict[str, Any] = {"threshold": self._threshold}
            if self._model:
                kwargs["model"] = self._model
            self._metrics.append(cls(**kwargs))
        self._initialised = True

    async def grade(self, task: EvalTask) -> GraderResult:
        """Run deepeval metrics on a single task."""
        self._ensure_initialised()
        from deepeval.test_case import LLMTestCase

        response = task.metadata.get("response", "")
        contexts = task.metadata.get("contexts", [])
        if not contexts and task.context:
            contexts = [task.context]

        test_case = LLMTestCase(
            input=task.query,
            actual_output=response,
            expected_output=task.expected_answer or None,
            retrieval_context=contexts or None,
        )

        dimensions: dict[str, float] = {}
        reasons: list[str] = []

        for metric in self._metrics:
            metric.measure(test_case)
            score = float(metric.score) if metric.score is not None else 0.0
            dimensions[metric.__class__.__name__] = score
            if metric.reason:
                reasons.append(f"{metric.__class__.__name__}: {metric.reason}")

        avg_score = sum(dimensions.values()) / len(dimensions) if dimensions else 0.0
        is_correct = avg_score >= self._threshold

        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            is_correct=is_correct,
            score=avg_score,
            reasoning="; ".join(reasons) if reasons else "no reasoning",
            dimensions=dimensions,
        )
