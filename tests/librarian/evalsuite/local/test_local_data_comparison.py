"""Variant comparison against real eval data (local only — never committed).

Reads golden queries from a JSONL file on disk via the EVAL_DATASET_PATH env var.
Skips automatically if the env var is not set or the file doesn't exist, so CI
always passes cleanly.

Usage:
    EVAL_DATASET_PATH=/path/to/eval_dataset.jsonl \
        uv run pytest tests/librarian/evalsuite/local/ -v -s

Results with InMemoryRetriever (default, no infrastructure needed)
-------------------------------------------------------------------
Retrieval is done with a MockEmbedder over an empty in-process store.
Hit rates will be ~0% because no real corpus is indexed — this mode
validates the loader + eval pipeline with real query shapes, not actual
retrieval quality.

Results with real OpenSearch (meaningful retrieval scores)
----------------------------------------------------------
Set OPENSEARCH_URL + OPENSEARCH_INDEX to point at a live index that has the
cs_agent corpus ingested, then swap ``_make_retrieve_fn`` to use
OpenSearchRetriever with the variant's config.  See TODO(4) below.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from eval.loaders import load_golden_from_jsonl
from eval.metrics.retrieval_eval import evaluate_retrieval
from eval.variants import VARIANTS
from librarian.schemas.retrieval import RetrievalResult
from librarian.tasks.models import GoldenSample, RetrievalMetrics
from storage.vectordb.inmemory import InMemoryRetriever
from tests.librarian.testing.mock_embedder import MockEmbedder

# ---------------------------------------------------------------------------
# Path resolution — controlled by env var, never hardcoded
# ---------------------------------------------------------------------------

_EVAL_DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "")
_DATA_AVAILABLE = bool(_EVAL_DATASET_PATH) and Path(_EVAL_DATASET_PATH).exists()

pytestmark = pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason=(
        "EVAL_DATASET_PATH not set or file not found — "
        "set EVAL_DATASET_PATH=/abs/path/to/eval_dataset.jsonl to run"
    ),
)

# ---------------------------------------------------------------------------
# Dataset fixture — loaded once, shared across tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_golden() -> list[GoldenSample]:
    """Load golden samples from the external JSONL file."""
    samples = load_golden_from_jsonl(_EVAL_DATASET_PATH)
    print(f"\n  Loaded {len(samples)} unique golden samples from {_EVAL_DATASET_PATH}")
    by_difficulty = {}
    for s in samples:
        by_difficulty[s.difficulty] = by_difficulty.get(s.difficulty, 0) + 1
    print(f"  Difficulty breakdown: {by_difficulty}")
    return samples


# ---------------------------------------------------------------------------
# Retrieve function factory
#
# TODO(4): To run against real OpenSearch, replace _make_inmemory_retrieve_fn
# with a function that builds OpenSearchRetriever(cfg) and calls .search().
# Example:
#
#   async def _make_opensearch_retrieve_fn(cfg):
#       from librarian.retrieval.opensearch import OpenSearchRetriever
#       retriever = OpenSearchRetriever(
#           url=os.environ["OPENSEARCH_URL"],
#           index=os.environ["OPENSEARCH_INDEX"],
#           bm25_weight=cfg.bm25_weight,
#           vector_weight=cfg.vector_weight,
#       )
#       embedder = MiniLMEmbedder(cfg.embedding_model)  # or MultilingualEmbedder
#       async def retrieve_fn(query: str) -> list[RetrievalResult]:
#           vec = await embedder.aembed_query(query)
#           return await retriever.search(query, vec, k=cfg.retrieval_k)
#       return retrieve_fn
# ---------------------------------------------------------------------------


def _make_inmemory_retrieve_fn(cfg, embedder: MockEmbedder):
    """Retrieve against an empty InMemoryRetriever.

    Hit rates will be ~0% — no real corpus is indexed.
    This validates pipeline plumbing with real query shapes.
    """
    retriever = InMemoryRetriever(
        bm25_weight=cfg.bm25_weight,
        vector_weight=cfg.vector_weight,
    )

    async def retrieve_fn(query: str) -> list[RetrievalResult]:
        vec = embedder.embed_query(query)
        return await retriever.search(query_text=query, query_vector=vec, k=cfg.retrieval_k)

    return retrieve_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("variant_name", list(VARIANTS.keys()))
async def test_variant_on_real_data(
    variant_name: str,
    real_golden: list[GoldenSample],
) -> None:
    """Run eval_dataset.jsonl queries through each retrieval variant.

    With InMemoryRetriever (default): validates pipeline plumbing, not retrieval quality.
    With real OpenSearch: produces meaningful hit_rate and MRR scores.
    """
    cfg = VARIANTS[variant_name]
    embedder = MockEmbedder(dim=64, seed=42)
    retrieve_fn = _make_inmemory_retrieve_fn(cfg, embedder)

    metrics, clusters = await evaluate_retrieval(
        real_golden, retrieve_fn, k=cfg.retrieval_k
    )
    _print_result(variant_name, metrics, clusters)


def _print_result(variant_name: str, metrics: RetrievalMetrics, clusters: list) -> None:
    failure_summary = (
        ", ".join(f"{c.failure_type}×{c.count}" for c in clusters[:3]) or "none"
    )
    print(
        f"\n  [{variant_name:10s}]"
        f"  hit_rate@{metrics.k}={metrics.hit_rate_at_k:.3f}"
        f"  mrr={metrics.mrr:.3f}"
        f"  n={metrics.n_queries}"
        f"  failures=[{failure_summary}]"
    )


# ---------------------------------------------------------------------------
# Loader smoke test — always runnable (does not need retrieve_fn)
# ---------------------------------------------------------------------------


def test_loader_field_mapping() -> None:
    """Verify GoldenSample fields map correctly from the JSONL schema."""
    samples = load_golden_from_jsonl(_EVAL_DATASET_PATH)
    sample = samples[0]

    assert sample.query_id, "query_id must be non-empty"
    assert sample.query, "query must be non-empty"
    assert sample.expected_doc_url, "expected_doc_url must map from source_doc_id"
    assert sample.language == "de", "all cs_agent queries are German"
    assert sample.difficulty in ("easy", "medium", "hard")
    assert isinstance(sample.relevant_chunk_ids, list)
