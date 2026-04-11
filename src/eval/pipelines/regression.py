"""Regression evaluation pipeline.

Runs golden traces through retrieval, computes hit_rate@k and MRR,
compares against threshold floors, and optionally populates a snippet
store for FAQ bypass.

Flow: golden tasks -> retrieve_fn -> hit/MRR metrics -> threshold check -> EvalReport
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

import structlog

from eval.models import (
    CategoryBreakdown,
    EvalReport,
    EvalRunConfig,
    EvalTask,
    GraderResult,
)
from librarian.tasks.tracing import FailureClusterer, PipelineTracer

log = structlog.get_logger(__name__)

RetrieveFn = Callable[[str], Coroutine[Any, Any, list[Any]]]


class RegressionThresholds:
    """Metric floors for regression detection."""

    def __init__(
        self,
        *,
        hit_rate_floor: float = 0.6,
        mrr_floor: float = 0.4,
    ) -> None:
        self.hit_rate_floor = hit_rate_floor
        self.mrr_floor = mrr_floor


async def run_regression_eval(
    tasks: list[EvalTask],
    retrieve_fn: RetrieveFn,
    *,
    k: int = 5,
    url_extractor: Callable[[Any], str] | None = None,
    thresholds: RegressionThresholds | None = None,
    config: EvalRunConfig | None = None,
    clusterer: FailureClusterer | None = None,
) -> EvalReport:
    """Evaluate retrieval quality against golden tasks.

    Args:
        tasks: Golden eval tasks. ``expected_answer`` or ``metadata["expected_doc_url"]``
               should contain the expected URL.
        retrieve_fn: Async callable ``(query) -> list[results]``.
        k: Cutoff for hit-rate and MRR.
        url_extractor: Callable to extract URL from a retrieval result.
                       Defaults to ``lambda r: r.chunk.metadata.url``.
        thresholds: Metric floors for pass/fail.
        config: Optional run configuration.
        clusterer: Optional failure clusterer.

    Returns:
        EvalReport with retrieval metrics and failure clusters.
    """
    if not tasks:
        msg = "Task list is empty — nothing to evaluate"
        raise ValueError(msg)

    config = config or EvalRunConfig()
    thresholds = thresholds or RegressionThresholds()
    clusterer = clusterer or FailureClusterer()
    url_fn = url_extractor or _default_url_extractor

    tracer = PipelineTracer()
    hits: list[int] = []
    reciprocal_ranks: list[float] = []
    results: list[GraderResult] = []

    for task in tasks:
        trace = tracer.create_trace(task.id, task.query)
        retrieved = await retrieve_fn(task.query)

        expected_url = task.metadata.get("expected_doc_url", task.expected_answer)
        urls = [url_fn(r) for r in retrieved[:k]]

        hit = expected_url in urls
        hits.append(int(hit))

        rr = next(
            (1.0 / (i + 1) for i, u in enumerate(urls) if u == expected_url),
            0.0,
        )
        reciprocal_ranks.append(rr)

        trace.status = "success" if hit else "failure"
        trace.confidence = rr
        if not hit:
            trace.failure_reason = "expected_doc_not_in_top_k"

        results.append(
            GraderResult(
                task_id=task.id,
                grader_type="retrieval_regression",
                is_correct=hit,
                score=rr,
                reasoning=f"hit={'yes' if hit else 'no'}, rr={rr:.3f}",
                dimensions={"reciprocal_rank": rr},
            )
        )

    n = len(tasks)
    hit_rate = sum(hits) / n
    mrr = sum(reciprocal_ranks) / n

    clusters = clusterer.cluster_failures(tracer.get_failure_traces())

    log.info("regression.done", hit_rate=hit_rate, mrr=mrr, k=k, n=n)

    meets_thresholds = (
        hit_rate >= thresholds.hit_rate_floor and mrr >= thresholds.mrr_floor
    )

    return EvalReport(
        run_id=config.run_name or "regression_eval",
        config=config,
        results=results,
        pass_rate=hit_rate,
        avg_score=mrr,
        n_tasks=n,
        n_passed=sum(hits),
        by_category=[
            CategoryBreakdown(
                category="hit_rate@k",
                n_tasks=n,
                pass_rate=hit_rate,
                avg_score=hit_rate,
            ),
            CategoryBreakdown(
                category="mrr",
                n_tasks=n,
                pass_rate=1.0 if mrr >= thresholds.mrr_floor else 0.0,
                avg_score=mrr,
            ),
        ],
        failure_clusters=[
            {"type": c.failure_type, "count": c.count, "patterns": c.common_patterns}
            for c in clusters
        ],
    )


def _default_url_extractor(result: Any) -> str:
    """Default URL extractor — assumes librarian-style RetrievalResult."""
    try:
        return result.chunk.metadata.url
    except AttributeError:
        return str(result)
