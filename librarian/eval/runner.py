"""EvalRunner — orchestrates harnesses + per-stage metrics.

Central entry point for running evaluations.  Composes the capability
and regression harnesses with optional per-stage metric collection
(retrieval, reranker, confidence gate).
"""

from __future__ import annotations

from typing import Any

import structlog

from eval.harnesses.capability import run_capability_eval
from eval.harnesses.regression import (
    RegressionThresholds,
    run_regression_eval,
)
from eval.metrics._shared import RetrieveFn
from eval.metrics.confidence import GateMetrics, evaluate_gate
from eval.metrics.reranker import RerankerMetrics, evaluate_reranker
from eval.models import EvalReport, EvalRunConfig, EvalTask
from librarian.schemas.chunks import GradedChunk, RankedChunk
from librarian.ingestion.tasks.tracing import FailureClusterer

log = structlog.get_logger(__name__)


class EvalRunner:
    """Configurable evaluation orchestrator.

    Composes harnesses (capability, regression) with optional per-stage
    metrics (reranker quality, confidence gate calibration).

    Args:
        graders: List of grader instances implementing the Grader protocol.
        config: Run configuration for reproducibility.
        clusterer: Optional failure clusterer override.
    """

    def __init__(
        self,
        graders: list[Any],
        *,
        config: EvalRunConfig | None = None,
        clusterer: FailureClusterer | None = None,
    ) -> None:
        self._graders = graders
        self._config = config or EvalRunConfig()
        self._clusterer = clusterer

    async def run_capability(self, tasks: list[EvalTask]) -> EvalReport:
        """Run capability evaluation harness."""
        log.info(
            "runner.capability.start",
            n_tasks=len(tasks),
            n_graders=len(self._graders),
        )
        return await run_capability_eval(
            tasks,
            self._graders,
            config=self._config,
            clusterer=self._clusterer,
        )

    async def run_regression(
        self,
        tasks: list[EvalTask],
        retrieve_fn: RetrieveFn,
        *,
        k: int = 5,
        url_extractor: Any | None = None,
        thresholds: RegressionThresholds | None = None,
    ) -> EvalReport:
        """Run regression evaluation harness."""
        log.info("runner.regression.start", n_tasks=len(tasks), k=k)
        return await run_regression_eval(
            tasks,
            retrieve_fn,
            k=k,
            url_extractor=url_extractor,
            thresholds=thresholds,
            config=self._config,
            clusterer=self._clusterer,
        )

    @staticmethod
    def evaluate_reranker(
        queries: list[tuple[list[GradedChunk], list[RankedChunk], set[str]]],
        k: int = 5,
    ) -> RerankerMetrics:
        """Evaluate reranker quality across multiple queries."""
        return evaluate_reranker(queries, k=k)

    @staticmethod
    def evaluate_gate(
        scores: list[float],
        truths: list[bool],
        threshold: float = 0.3,
    ) -> GateMetrics:
        """Evaluate confidence gate calibration."""
        return evaluate_gate(scores, truths, threshold)
