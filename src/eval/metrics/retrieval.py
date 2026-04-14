"""Retrieval evaluation metrics.

Computes hit_rate@k, MRR, precision@k, recall@k, and NDCG@k over a
golden dataset.  Traces each query and clusters failures with the
RAG-aware FailureClusterer.

Uses the shared core loop from ``eval.metrics._shared`` to avoid
duplicating hit/MRR logic with ``eval.harnesses.regression``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Coroutine
from typing import Any

from core.logging import get_logger
from eval.metrics._shared import (
    RetrievalHit,
    aggregate_hit_rate,
    aggregate_mrr,
    compute_retrieval_hits,
)
from librarian.schemas.retrieval import RetrievalResult
from librarian.tasks.models import GoldenSample, RetrievalMetrics
from librarian.tasks.tracing import (
    FailureCluster,
    FailureClusterer,
    PipelineTracer,
)

log = get_logger(__name__)

RetrieveFn = Callable[[str], Coroutine[Any, Any, list[RetrievalResult]]]


# ---------------------------------------------------------------------------
# LangFuse helpers
# ---------------------------------------------------------------------------


def _log_langfuse_scores(metrics: RetrievalMetrics, trace_id: str) -> None:
    """Log retrieval metrics as LangFuse scores.  No-op if LangFuse is unconfigured."""
    try:
        from langfuse import Langfuse

        lf = Langfuse()
        lf.create_score(
            trace_id=trace_id, name="hit_rate_at_k", value=metrics.hit_rate_at_k
        )
        lf.create_score(trace_id=trace_id, name="mrr", value=metrics.mrr)
        log.info("eval.langfuse.scores.logged", trace_id=trace_id)
    except Exception as exc:
        log.warning("eval.langfuse.scores.failed", error=str(exc))


# ---------------------------------------------------------------------------
# Accessors for GoldenSample → shared core
# ---------------------------------------------------------------------------


def _golden_url_extractor(result: RetrievalResult) -> str:
    return result.chunk.metadata.url


# ---------------------------------------------------------------------------
# Core evaluation entry point
# ---------------------------------------------------------------------------


async def evaluate_retrieval(
    golden: list[GoldenSample],
    retrieve_fn: RetrieveFn,
    k: int = 5,
    langfuse_trace_id: str | None = None,
) -> tuple[RetrievalMetrics, list[FailureCluster]]:
    """Evaluate retrieval quality against a golden dataset.

    Args:
        golden:      List of golden samples (query + expected_doc_url).
        retrieve_fn: Async callable ``(query: str) -> list[RetrievalResult]``.
        k:           Cutoff for hit-rate and MRR calculation.
        langfuse_trace_id: Optional trace ID for LangFuse score logging.

    Returns:
        Tuple of (RetrievalMetrics, list[FailureCluster]).
    """
    if not golden:
        raise ValueError("golden dataset is empty — nothing to evaluate")

    hits = await compute_retrieval_hits(
        golden,
        retrieve_fn,
        k,
        id_fn=lambda s: s.query_id,
        query_fn=lambda s: s.query,
        expected_url_fn=lambda s: s.expected_doc_url,
        url_extractor=_golden_url_extractor,
    )

    # Trace failures for clustering
    tracer = PipelineTracer()
    for h in hits:
        trace = tracer.create_trace(h.task_id, h.query)
        trace.status = "success" if h.hit else "failure"
        trace.confidence = h.reciprocal_rank
        trace.failure_reason = None if h.hit else "expected_doc_not_in_top_k"

    clusterer = FailureClusterer()
    clusters = clusterer.cluster_failures(tracer.get_failure_traces())

    hit_rate = aggregate_hit_rate(hits)
    mrr = aggregate_mrr(hits)

    metrics = RetrievalMetrics(
        hit_rate_at_k=hit_rate,
        mrr=mrr,
        k=k,
        n_queries=len(golden),
    )
    log.info("eval.retrieval.done", **metrics.model_dump())
    log.info("eval.failure_clusters", clusters=clusterer.get_summary())
    if langfuse_trace_id:
        _log_langfuse_scores(metrics, langfuse_trace_id)
    return metrics, clusters


# ---------------------------------------------------------------------------
# Extended retrieval metrics (precision, recall, NDCG)
# ---------------------------------------------------------------------------


def precision_at_k(
    hits: list[RetrievalHit],
    relevant_urls_map: dict[str, set[str]],
    k: int,
) -> float:
    """Fraction of top-k results that are relevant, averaged over queries.

    Args:
        hits: Retrieval hits from ``compute_retrieval_hits``.
        relevant_urls_map: ``{task_id: set(relevant_urls)}`` ground truth.
        k: Cutoff (should match the k used in compute_retrieval_hits).
    """
    if not hits:
        return 0.0
    precisions = []
    for h in hits:
        relevant = relevant_urls_map.get(h.task_id, set())
        n_relevant_in_k = sum(1 for url in h.retrieved_urls[:k] if url in relevant)
        precisions.append(n_relevant_in_k / k)
    return sum(precisions) / len(precisions)


def recall_at_k(
    hits: list[RetrievalHit],
    relevant_urls_map: dict[str, set[str]],
    k: int,
) -> float:
    """Fraction of all relevant docs found in top-k, averaged over queries.

    Args:
        hits: Retrieval hits from ``compute_retrieval_hits``.
        relevant_urls_map: ``{task_id: set(relevant_urls)}`` ground truth.
        k: Cutoff.
    """
    if not hits:
        return 0.0
    recalls = []
    for h in hits:
        relevant = relevant_urls_map.get(h.task_id, set())
        if not relevant:
            recalls.append(0.0)
            continue
        n_relevant_in_k = sum(1 for url in h.retrieved_urls[:k] if url in relevant)
        recalls.append(n_relevant_in_k / len(relevant))
    return sum(recalls) / len(recalls)


def ndcg_at_k(
    hits: list[RetrievalHit],
    relevant_urls_map: dict[str, set[str]],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at k, averaged over queries.

    Uses binary relevance: 1 if URL is in the relevant set, 0 otherwise.

    Args:
        hits: Retrieval hits from ``compute_retrieval_hits``.
        relevant_urls_map: ``{task_id: set(relevant_urls)}`` ground truth.
        k: Cutoff.
    """
    if not hits:
        return 0.0
    ndcgs = []
    for h in hits:
        relevant = relevant_urls_map.get(h.task_id, set())
        dcg = sum(
            (1.0 / math.log2(i + 2))
            for i, url in enumerate(h.retrieved_urls[:k])
            if url in relevant
        )
        n_relevant = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(ndcgs) / len(ndcgs)
