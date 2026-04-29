"""Shared retrieval evaluation primitives.

Provides the core hit/MRR computation loop used by both
``eval.metrics.retrieval`` and ``eval.harnesses.regression``.
Generic over input type via accessor callables.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")
RetrieveFn = Callable[[str], Coroutine[Any, Any, list[Any]]]


@dataclass
class RetrievalHit:
    """Result of evaluating a single query against a retrieval function."""

    task_id: str
    query: str
    hit: bool
    reciprocal_rank: float
    retrieved_urls: list[str] = field(default_factory=list)
    expected_url: str = ""


async def compute_retrieval_hits(
    items: Sequence[T],
    retrieve_fn: RetrieveFn,
    k: int,
    *,
    id_fn: Callable[[T], str],
    query_fn: Callable[[T], str],
    expected_url_fn: Callable[[T], str],
    url_extractor: Callable[[Any], str],
) -> list[RetrievalHit]:
    """Run *retrieve_fn* per item and compute hit + reciprocal rank.

    This is the shared inner loop that both the metrics module and the
    regression harness delegate to.  It is generic over input type *T*
    via accessor callables so it can work with ``GoldenSample``,
    ``EvalTask``, or any future evaluation item.

    Args:
        items:          Sequence of evaluation items.
        retrieve_fn:    Async callable ``(query: str) -> list[results]``.
        k:              Cutoff for hit-rate and MRR calculation.
        id_fn:          Extract an identifier from an item.
        query_fn:       Extract the query string from an item.
        expected_url_fn: Extract the expected URL from an item.
        url_extractor:  Extract a URL string from a single retrieval result.

    Returns:
        List of ``RetrievalHit`` — one per input item.
    """
    hits: list[RetrievalHit] = []

    for item in items:
        task_id = id_fn(item)
        query = query_fn(item)
        expected_url = expected_url_fn(item)

        results = await retrieve_fn(query)
        urls = [url_extractor(r) for r in results[:k]]

        hit = expected_url in urls
        rr = next(
            (1.0 / (i + 1) for i, u in enumerate(urls) if u == expected_url),
            0.0,
        )

        hits.append(
            RetrievalHit(
                task_id=task_id,
                query=query,
                hit=hit,
                reciprocal_rank=rr,
                retrieved_urls=urls,
                expected_url=expected_url,
            )
        )

    return hits


def aggregate_hit_rate(hits: list[RetrievalHit]) -> float:
    """Compute hit-rate from a list of retrieval hits."""
    if not hits:
        return 0.0
    return sum(1 for h in hits if h.hit) / len(hits)


def aggregate_mrr(hits: list[RetrievalHit]) -> float:
    """Compute Mean Reciprocal Rank from a list of retrieval hits."""
    if not hits:
        return 0.0
    return sum(h.reciprocal_rank for h in hits) / len(hits)
