"""Regression evaluation harness.

Runs golden traces through retrieval, computes hit_rate@k and MRR via
the shared metrics core, compares against threshold floors, and wraps
results into an ``EvalReport``.

Flow: golden tasks -> retrieve_fn -> shared hit/MRR core -> threshold check -> EvalReport

FAQ slice: :func:`~evals.utils.loaders.load_golden_from_faq_csv` (``limit=50``) and
:func:`~evals.utils.loaders.golden_samples_to_eval_tasks` — see ``tests/test_regression_faq.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

from evals.metrics._shared import (
    RetrieveFn,
    aggregate_hit_rate,
    aggregate_mrr,
    compute_retrieval_hits,
)
from evals.utils.models import (
    CategoryBreakdown,
    EvalReport,
    EvalRunConfig,
    EvalTask,
    GraderResult,
)
from evals.utils.tracing import FailureClusterer, PipelineTracer

log = structlog.get_logger(__name__)


def _default_url_extractor(result: Any) -> str:
    """Default URL extractor — assumes librarian-style RetrievalResult."""
    try:
        return result.chunk.metadata.url
    except AttributeError:
        return str(result)


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

    hits = await compute_retrieval_hits(
        tasks,
        retrieve_fn,
        k,
        id_fn=lambda t: t.id,
        query_fn=lambda t: t.query,
        expected_url_fn=lambda t: t.metadata.get("expected_doc_url", t.expected_answer),
        url_extractor=url_fn,
    )

    # Trace failures for clustering
    tracer = PipelineTracer()
    results: list[GraderResult] = []
    for h in hits:
        trace = tracer.create_trace(h.task_id, h.query)
        trace.status = "success" if h.hit else "failure"
        trace.confidence = h.reciprocal_rank
        if not h.hit:
            trace.failure_reason = "expected_doc_not_in_top_k"

        results.append(
            GraderResult(
                task_id=h.task_id,
                grader_type="retrieval_regression",
                is_correct=h.hit,
                score=h.reciprocal_rank,
                reasoning=f"hit={'yes' if h.hit else 'no'}, rr={h.reciprocal_rank:.3f}",
                dimensions={"reciprocal_rank": h.reciprocal_rank},
            )
        )

    n = len(tasks)
    hit_rate = aggregate_hit_rate(hits)
    mrr = aggregate_mrr(hits)

    clusters = clusterer.cluster_failures(tracer.get_failure_traces())

    log.info("regression.done", hit_rate=hit_rate, mrr=mrr, k=k, n=n)

    return EvalReport(
        run_id=config.run_name or "regression_eval",
        config=config,
        results=results,
        pass_rate=hit_rate,
        avg_score=mrr,
        n_tasks=n,
        n_passed=sum(1 for h in hits if h.hit),
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
