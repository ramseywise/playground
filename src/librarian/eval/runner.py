"""EvalRunner — orchestrates tasks + graders into an EvalReport.

Central entry point for running evaluations.  Supports both capability
and regression pipelines with configurable grader sets.
"""

from __future__ import annotations

from typing import Any

import structlog

from agents.librarian.eval.models import EvalReport, EvalRunConfig, EvalTask
from agents.librarian.eval.pipelines.capability import run_capability_eval
from agents.librarian.eval.pipelines.regression import (
    RegressionThresholds,
    RetrieveFn,
    run_regression_eval,
)
from agents.librarian.eval.tasks.tracing import FailureClusterer

log = structlog.get_logger(__name__)


class EvalRunner:
    """Configurable evaluation orchestrator.

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
        """Run capability evaluation pipeline."""
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
        """Run regression evaluation pipeline."""
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
