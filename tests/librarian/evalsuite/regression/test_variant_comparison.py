"""Side-by-side retrieval comparison across named configuration variants.

Runs the same golden dataset against three retrieval configurations:
  librarian — hybrid BM25+dense, multilingual-e5-large, CrossEncoder, k=10
  raptor    — pure knn, minilm, no reranker, k=5 (cs_agent_assist_with_rag params)
  bedrock   — AWS out-of-the-box approximation, pure knn, no reranker, k=5

This test does NOT assert metric floors — use test_retrieval_metrics.py for that.
It is a comparison harness: run it to see relative performance before/after changes.

Run a focused comparison:
    uv run pytest tests/librarian/evalsuite/regression/test_variant_comparison.py -v -s

The ``-s`` flag shows the printed comparison table.

Note on InMemoryRetriever:
  bm25_weight / vector_weight are stored in each variant's LibrarySettings but
  InMemoryRetriever always applies equal-weight RRF fusion in tests — weights only
  take effect in production OpenSearch.  The main differentiators in tests are
  retrieval_k (5 vs 10) and reranker_strategy.
"""

from __future__ import annotations

import pytest

from eval.metrics.retrieval import evaluate_retrieval
from eval.variants import VARIANTS
from librarian.schemas.chunks import Chunk
from librarian.schemas.retrieval import RetrievalResult
from librarian.ingestion.tasks.models import GoldenSample, RetrievalMetrics
from storage.vectordb.inmemory import InMemoryRetriever
from tests.librarian.evalsuite.conftest import CORPUS, GOLDEN
from tests.librarian.testing.mock_embedder import MockEmbedder


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def golden_samples() -> list[GoldenSample]:
    return GOLDEN


async def _build_populated_retriever(
    corpus: list[Chunk],
    bm25_weight: float,
    vector_weight: float,
    embedder: MockEmbedder,
) -> InMemoryRetriever:
    """Index corpus into a fresh InMemoryRetriever with per-variant weights."""
    retriever = InMemoryRetriever(bm25_weight=bm25_weight, vector_weight=vector_weight)
    chunks_with_embeddings = [
        chunk.model_copy(update={"embedding": embedder.embed_passage(chunk.text)})
        for chunk in corpus
    ]
    await retriever.upsert(chunks_with_embeddings)
    return retriever


# ---------------------------------------------------------------------------
# Comparison: one parametrized run per variant, prints side-by-side results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("variant_name", list(VARIANTS.keys()))
async def test_variant_retrieval_comparison(
    variant_name: str,
    golden_samples: list[GoldenSample],
) -> None:
    """Evaluate retrieval quality for a named configuration variant.

    Each variant is evaluated at its own retrieval_k so results reflect
    production conditions (raptor/bedrock retrieve 5, librarian retrieves 10).
    """
    cfg = VARIANTS[variant_name]
    if cfg.retrieval_strategy == "bedrock" and not cfg.bedrock_knowledge_base_id:
        pytest.skip("BEDROCK_KNOWLEDGE_BASE_ID not set")
    if cfg.retrieval_strategy == "google_adk" and not (
        cfg.google_datastore_id or cfg.google_project_id
    ):
        pytest.skip("GOOGLE_DATASTORE_ID not set")
    embedder = MockEmbedder(dim=64, seed=42)
    retriever = await _build_populated_retriever(
        corpus=CORPUS,
        bm25_weight=cfg.bm25_weight,
        vector_weight=cfg.vector_weight,
        embedder=embedder,
    )
    k = cfg.retrieval_k

    async def retrieve_fn(query: str) -> list[RetrievalResult]:
        vec = embedder.embed_query(query)
        return await retriever.search(query_text=query, query_vector=vec, k=k)

    metrics, clusters = await evaluate_retrieval(golden_samples, retrieve_fn, k=k)
    _print_variant_result(variant_name, metrics, clusters)


def _print_variant_result(
    variant_name: str,
    metrics: RetrievalMetrics,
    clusters: list,
) -> None:
    failure_summary = (
        ", ".join(f"{c.failure_type}×{c.count}" for c in clusters) or "none"
    )
    print(
        f"\n  [{variant_name:10s}]"
        f"  hit_rate@{metrics.k}={metrics.hit_rate_at_k:.3f}"
        f"  mrr={metrics.mrr:.3f}"
        f"  n={metrics.n_queries}"
        f"  failures=[{failure_summary}]"
    )


# ---------------------------------------------------------------------------
# Sanity: all variants return valid metric shapes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("variant_name", list(VARIANTS.keys()))
async def test_variant_metrics_are_valid(
    variant_name: str,
    golden_samples: list[GoldenSample],
) -> None:
    """All variants must return well-formed RetrievalMetrics — no exceptions, valid range."""
    cfg = VARIANTS[variant_name]
    if cfg.retrieval_strategy == "bedrock" and not cfg.bedrock_knowledge_base_id:
        pytest.skip("BEDROCK_KNOWLEDGE_BASE_ID not set")
    if cfg.retrieval_strategy == "google_adk" and not (
        cfg.google_datastore_id or cfg.google_project_id
    ):
        pytest.skip("GOOGLE_DATASTORE_ID not set")
    embedder = MockEmbedder(dim=64, seed=42)
    retriever = await _build_populated_retriever(
        corpus=CORPUS,
        bm25_weight=cfg.bm25_weight,
        vector_weight=cfg.vector_weight,
        embedder=embedder,
    )
    k = cfg.retrieval_k

    async def retrieve_fn(query: str) -> list[RetrievalResult]:
        vec = embedder.embed_query(query)
        return await retriever.search(query_text=query, query_vector=vec, k=k)

    metrics, _ = await evaluate_retrieval(golden_samples, retrieve_fn, k=k)

    assert isinstance(metrics, RetrievalMetrics)
    assert metrics.n_queries == len(golden_samples)
    assert metrics.k == k
    assert 0.0 <= metrics.hit_rate_at_k <= 1.0
    assert 0.0 <= metrics.mrr <= 1.0
