"""Reranker evaluation metrics.

Measures how much the reranker improves retrieval quality by comparing
pre-rerank (``GradedChunk``) and post-rerank (``RankedChunk``) orderings
against ground-truth relevance judgements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from librarian.schemas.chunks import GradedChunk, RankedChunk


@dataclass
class RerankerMetrics:
    """Aggregate metrics for a single reranker evaluation run."""

    avg_rank_displacement: float = 0.0
    ndcg_before: float = 0.0
    ndcg_after: float = 0.0
    ndcg_improvement: float = 0.0
    precision_lift_at_k: float = 0.0
    n_queries: int = 0


# ---------------------------------------------------------------------------
# Per-query helpers
# ---------------------------------------------------------------------------


def _binary_relevance(chunk_id: str, relevant_ids: set[str]) -> float:
    return 1.0 if chunk_id in relevant_ids else 0.0


def _dcg(relevances: list[float]) -> float:
    """Discounted Cumulative Gain with log2 discount."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def _ndcg(relevances: list[float], n_relevant: int) -> float:
    """Normalized DCG.  *n_relevant* is the total relevant count for ideal DCG."""
    dcg = _dcg(relevances)
    ideal = _dcg([1.0] * min(n_relevant, len(relevances)))
    return dcg / ideal if ideal > 0 else 0.0


def _precision_at_k(chunk_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = chunk_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for c in top_k if c in relevant_ids) / k


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rank_displacement(
    before: list[GradedChunk],
    after: list[RankedChunk],
) -> float:
    """Average absolute position change per chunk after reranking.

    Higher values mean the reranker moved chunks more aggressively.
    Returns 0.0 if inputs are empty.
    """
    if not before or not after:
        return 0.0

    before_order = {c.chunk.id: i for i, c in enumerate(before)}
    after_order = {c.chunk.id: i for i, c in enumerate(after)}

    common = set(before_order) & set(after_order)
    if not common:
        return 0.0
    return sum(abs(before_order[cid] - after_order[cid]) for cid in common) / len(common)


def ndcg_improvement(
    before: list[GradedChunk],
    after: list[RankedChunk],
    relevant_chunk_ids: set[str],
    k: int = 5,
) -> tuple[float, float, float]:
    """NDCG@k before and after reranking, plus the delta.

    Returns:
        Tuple of (ndcg_before, ndcg_after, improvement).
    """
    n_relevant = len(relevant_chunk_ids)

    before_rels = [
        _binary_relevance(c.chunk.id, relevant_chunk_ids) for c in before[:k]
    ]
    after_rels = [
        _binary_relevance(c.chunk.id, relevant_chunk_ids) for c in after[:k]
    ]

    ndcg_b = _ndcg(before_rels, n_relevant)
    ndcg_a = _ndcg(after_rels, n_relevant)
    return ndcg_b, ndcg_a, ndcg_a - ndcg_b


def score_correlation(
    reranker_scores: list[float],
    ground_truth_relevance: list[float],
) -> float:
    """Spearman rank correlation between reranker scores and ground truth.

    Uses a simplified rank-based calculation.  Returns 0.0 if fewer than
    2 items are provided.
    """
    n = len(reranker_scores)
    if n < 2 or len(ground_truth_relevance) != n:
        return 0.0

    def _ranks(values: list[float]) -> list[float]:
        indexed = sorted(enumerate(values), key=lambda x: x[1], reverse=True)
        ranks = [0.0] * n
        for rank, (orig_idx, _) in enumerate(indexed):
            ranks[orig_idx] = float(rank + 1)
        return ranks

    r1 = _ranks(reranker_scores)
    r2 = _ranks(ground_truth_relevance)
    d_sq = sum((a - b) ** 2 for a, b in zip(r1, r2))
    return 1.0 - (6.0 * d_sq) / (n * (n**2 - 1))


def top_k_precision_lift(
    before: list[GradedChunk],
    after: list[RankedChunk],
    relevant_chunk_ids: set[str],
    k: int = 5,
) -> float:
    """Precision@k improvement from reranking.

    Returns the delta: ``precision_after - precision_before``.
    """
    before_ids = [c.chunk.id for c in before]
    after_ids = [c.chunk.id for c in after]

    p_before = _precision_at_k(before_ids, relevant_chunk_ids, k)
    p_after = _precision_at_k(after_ids, relevant_chunk_ids, k)
    return p_after - p_before


def evaluate_reranker(
    queries: list[
        tuple[list[GradedChunk], list[RankedChunk], set[str]]
    ],
    k: int = 5,
) -> RerankerMetrics:
    """Evaluate reranker quality across multiple queries.

    Args:
        queries: List of ``(before, after, relevant_chunk_ids)`` tuples.
        k: Cutoff for precision and NDCG calculations.

    Returns:
        Aggregate ``RerankerMetrics``.
    """
    if not queries:
        return RerankerMetrics()

    displacements = []
    ndcg_befores = []
    ndcg_afters = []
    precision_lifts = []

    for before, after, relevant_ids in queries:
        displacements.append(rank_displacement(before, after))
        nb, na, _ = ndcg_improvement(before, after, relevant_ids, k)
        ndcg_befores.append(nb)
        ndcg_afters.append(na)
        precision_lifts.append(top_k_precision_lift(before, after, relevant_ids, k))

    n = len(queries)
    avg_before = sum(ndcg_befores) / n
    avg_after = sum(ndcg_afters) / n

    return RerankerMetrics(
        avg_rank_displacement=sum(displacements) / n,
        ndcg_before=avg_before,
        ndcg_after=avg_after,
        ndcg_improvement=avg_after - avg_before,
        precision_lift_at_k=sum(precision_lifts) / n,
        n_queries=n,
    )
