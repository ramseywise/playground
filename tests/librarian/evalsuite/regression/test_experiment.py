"""Tests for the LangFuse experiment runner.

Validates experiment logic (scoring, aggregation, comparison table)
without requiring a LangFuse connection.  All LangFuse calls are None
(LANGFUSE_ENABLED defaults to false in test settings).
"""

from __future__ import annotations

import pytest

from eval.experiment import (
    ExperimentResult,
    QueryResult,
    print_comparison_table,
    run_all_experiments,
    run_variant_experiment,
    upload_golden_dataset,
)
from eval.variants import VARIANTS
from librarian.config import LibrarySettings
from librarian.schemas.chunks import Chunk
from librarian.ingestion.tasks.models import GoldenSample
from tests.librarian.evalsuite.conftest import CORPUS, GOLDEN


# ---------------------------------------------------------------------------
# upload_golden_dataset — no-op without LangFuse
# ---------------------------------------------------------------------------


def test_upload_returns_zero_without_langfuse() -> None:
    """When LangFuse is not configured, upload returns 0."""
    n = upload_golden_dataset(GOLDEN, dataset_name="test_ds")
    assert n == 0


# ---------------------------------------------------------------------------
# run_variant_experiment — single variant
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_single_variant_returns_result() -> None:
    """Running a single variant produces a well-formed ExperimentResult."""
    result = await run_variant_experiment(
        "librarian",
        GOLDEN,
        CORPUS,
    )

    assert isinstance(result, ExperimentResult)
    assert result.variant_name == "librarian"
    assert result.n_queries == len(GOLDEN)
    assert 0.0 <= result.hit_rate <= 1.0
    assert 0.0 <= result.mrr <= 1.0
    assert result.avg_latency_ms >= 0.0
    assert len(result.query_results) == len(GOLDEN)
    assert result.config_snapshot["reranker_strategy"] == "cross_encoder"


@pytest.mark.asyncio
async def test_query_results_contain_expected_fields() -> None:
    """Each QueryResult has all required fields populated."""
    result = await run_variant_experiment(
        "librarian",
        GOLDEN,
        CORPUS,
    )

    for qr in result.query_results:
        assert isinstance(qr, QueryResult)
        assert qr.query_id
        assert qr.query
        assert qr.expected_url
        assert qr.latency_ms >= 0.0
        assert isinstance(qr.hit, bool)
        assert 0.0 <= qr.reciprocal_rank <= 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("variant_name", list(VARIANTS.keys()))
async def test_all_variants_produce_valid_results(variant_name: str) -> None:
    """Every registered variant runs without errors and produces valid metrics."""
    cfg = VARIANTS[variant_name]
    if cfg.retrieval_strategy == "bedrock" and not cfg.bedrock_knowledge_base_id:
        pytest.skip("BEDROCK_KNOWLEDGE_BASE_ID not set")
    if cfg.retrieval_strategy == "google_adk" and not (
        cfg.google_datastore_id or cfg.google_project_id
    ):
        pytest.skip("GOOGLE_DATASTORE_ID not set")

    result = await run_variant_experiment(
        variant_name,
        GOLDEN,
        CORPUS,
    )

    assert result.variant_name == variant_name
    assert result.n_queries == 5
    assert 0.0 <= result.hit_rate <= 1.0
    assert 0.0 <= result.mrr <= 1.0


# ---------------------------------------------------------------------------
# run_all_experiments — comparison across all variants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_all_returns_all_variants() -> None:
    """run_all_experiments returns results for every configured variant."""
    results = await run_all_experiments(GOLDEN, CORPUS)

    # Live variants are skipped when credentials are absent
    def _variant_runnable(name: str, cfg: LibrarySettings) -> bool:
        if cfg.retrieval_strategy == "bedrock" and not cfg.bedrock_knowledge_base_id:
            return False
        if cfg.retrieval_strategy == "google_adk" and not (
            cfg.google_datastore_id or cfg.google_project_id
        ):
            return False
        return True

    expected = {
        name for name, cfg in VARIANTS.items() if _variant_runnable(name, cfg)
    }
    assert set(results.keys()) == expected
    for name, result in results.items():
        assert result.variant_name == name
        assert result.n_queries == len(GOLDEN)


# ---------------------------------------------------------------------------
# ExperimentResult.summary_dict
# ---------------------------------------------------------------------------


def test_summary_dict_format() -> None:
    """summary_dict produces the expected fields for table rendering."""
    result = ExperimentResult(
        variant_name="test",
        dataset_name="ds",
        run_name="test_run",
        hit_rate=0.8,
        mrr=0.6,
        n_queries=10,
        n_hits=8,
        avg_latency_ms=1.5,
    )

    d = result.summary_dict()
    assert d["variant"] == "test"
    assert d["hit_rate"] == 0.8
    assert d["mrr"] == 0.6
    assert d["n_queries"] == 10
    assert d["n_hits"] == 8
    assert d["avg_latency_ms"] == 1.5
    assert isinstance(d["failure_types"], list)


# ---------------------------------------------------------------------------
# print_comparison_table — smoke test (doesn't crash)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_print_comparison_table_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """Comparison table prints without error."""
    results = await run_all_experiments(GOLDEN, CORPUS)
    print_comparison_table(results)

    captured = capsys.readouterr()
    assert "librarian" in captured.out
    assert "raptor" in captured.out
    # "bedrock" (mock) is always present; "bedrock-live" only when configured
    assert "bedrock" in captured.out
    assert "hit_rate" in captured.out
    assert "MRR" in captured.out


# ---------------------------------------------------------------------------
# Config snapshot captures variant-specific settings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_config_snapshot_differs_across_variants() -> None:
    """Each variant's config snapshot reflects its actual settings."""
    results = await run_all_experiments(GOLDEN, CORPUS)

    lib_snap = results["librarian"].config_snapshot
    rap_snap = results["raptor"].config_snapshot

    # Librarian uses cross_encoder, raptor uses passthrough
    assert lib_snap["reranker_strategy"] == "cross_encoder"
    assert rap_snap["reranker_strategy"] == "passthrough"

    # Librarian uses multilingual, raptor uses minilm
    assert lib_snap["embedding_provider"] == "multilingual"
    assert rap_snap["embedding_provider"] == "minilm"

    # Librarian retrieves 10, raptor retrieves 5
    assert lib_snap["retrieval_k"] == 10
    assert rap_snap["retrieval_k"] == 5
