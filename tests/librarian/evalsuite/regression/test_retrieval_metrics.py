"""Regression tests: retrieval hit_rate@k and MRR must not drop below thresholds.

These are not unit tests — they run the full retrieval pipeline against the
golden dataset and assert metric floors. Failure here means a regression in
retrieval quality, not a code bug.

Thresholds (conservative — green from day 1 with BM25-weighted InMemory):
    hit_rate@5 >= 0.6
    mrr        >= 0.4
"""

from __future__ import annotations

import pytest

from agents.librarian.rag_core.eval_harness.metrics.retrieval_eval import evaluate_retrieval
from agents.librarian.rag_core.eval_harness.tasks.models import GoldenSample, RetrievalMetrics
from agents.librarian.infra.storage.vectordb.inmemory import InMemoryRetriever
from agents.librarian.testing.mock_embedder import MockEmbedder
from agents.librarian.rag_core.schemas.retrieval import RetrievalResult

# Metric floors — update these (never lower them) when quality improves
HIT_RATE_FLOOR = 0.6
MRR_FLOOR = 0.4


# ---------------------------------------------------------------------------
# Regression: hit_rate@5
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hit_rate_meets_floor(
    golden_samples: list[GoldenSample],
    populated_retriever: InMemoryRetriever,
    eval_embedder: MockEmbedder,
) -> None:
    async def retrieve_fn(query: str) -> list[RetrievalResult]:
        vec = eval_embedder.embed_query(query)
        return await populated_retriever.search(query_text=query, query_vector=vec, k=5)

    metrics, _ = await evaluate_retrieval(golden_samples, retrieve_fn, k=5)
    assert metrics.hit_rate_at_k >= HIT_RATE_FLOOR, (
        f"hit_rate@5 regression: {metrics.hit_rate_at_k:.2f} < floor {HIT_RATE_FLOOR}"
    )


@pytest.mark.asyncio
async def test_mrr_meets_floor(
    golden_samples: list[GoldenSample],
    populated_retriever: InMemoryRetriever,
    eval_embedder: MockEmbedder,
) -> None:
    async def retrieve_fn(query: str) -> list[RetrievalResult]:
        vec = eval_embedder.embed_query(query)
        return await populated_retriever.search(query_text=query, query_vector=vec, k=5)

    metrics, _ = await evaluate_retrieval(golden_samples, retrieve_fn, k=5)
    assert metrics.mrr >= MRR_FLOOR, (
        f"MRR regression: {metrics.mrr:.2f} < floor {MRR_FLOOR}"
    )


# ---------------------------------------------------------------------------
# Regression: metric shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_shape(
    golden_samples: list[GoldenSample],
    populated_retriever: InMemoryRetriever,
    eval_embedder: MockEmbedder,
) -> None:
    async def retrieve_fn(query: str) -> list[RetrievalResult]:
        vec = eval_embedder.embed_query(query)
        return await populated_retriever.search(query_text=query, query_vector=vec, k=5)

    metrics, clusters = await evaluate_retrieval(golden_samples, retrieve_fn, k=5)

    assert isinstance(metrics, RetrievalMetrics)
    assert metrics.n_queries == len(golden_samples)
    assert metrics.k == 5
    assert 0.0 <= metrics.hit_rate_at_k <= 1.0
    assert 0.0 <= metrics.mrr <= 1.0
    assert isinstance(clusters, list)


# ---------------------------------------------------------------------------
# Regression: failure clustering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_catastrophic_failures(
    golden_samples: list[GoldenSample],
    populated_retriever: InMemoryRetriever,
    eval_embedder: MockEmbedder,
) -> None:
    """All 5 samples retrieved — zero_retrieval cluster should be absent."""

    async def retrieve_fn(query: str) -> list[RetrievalResult]:
        vec = eval_embedder.embed_query(query)
        return await populated_retriever.search(query_text=query, query_vector=vec, k=5)

    _, clusters = await evaluate_retrieval(golden_samples, retrieve_fn, k=5)
    failure_types = [c.failure_type for c in clusters]
    assert "zero_retrieval" not in failure_types


@pytest.mark.asyncio
async def test_empty_golden_raises() -> None:
    async def retrieve_fn(query: str) -> list[RetrievalResult]:
        return []

    with pytest.raises(ValueError, match="empty"):
        await evaluate_retrieval([], retrieve_fn, k=5)
