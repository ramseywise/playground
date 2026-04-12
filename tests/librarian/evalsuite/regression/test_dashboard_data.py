"""Tests for the eval dashboard data fetcher.

Validates JSON loading, export round-trip, and data model properties
without requiring a Langfuse connection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.dashboard_data import (
    DashboardData,
    FailureClusterData,
    QueryMetric,
    VariantResult,
    load_dashboard_data,
    load_from_dict,
    load_from_json,
    load_from_langfuse,
)
from eval.experiment import ExperimentResult, export_results, run_all_experiments
from tests.librarian.evalsuite.conftest import CORPUS, GOLDEN


# ---------------------------------------------------------------------------
# Fixture: a sample JSON export
# ---------------------------------------------------------------------------

SAMPLE_EXPORT: dict = {
    "exported_at": "2026-04-11T12:00:00",
    "variants": {
        "librarian": {
            "variant": "librarian",
            "run_name": "librarian_20260411T120000",
            "hit_rate": 0.8,
            "mrr": 0.65,
            "n_queries": 5,
            "n_hits": 4,
            "avg_latency_ms": 2.3,
            "dataset_name": "golden_eval",
            "config_snapshot": {
                "embedding_model": "intfloat/multilingual-e5-large",
                "embedding_provider": "multilingual",
                "reranker_strategy": "cross_encoder",
                "retrieval_k": 10,
                "bm25_weight": 0.3,
                "vector_weight": 0.7,
            },
            "query_results": [
                {
                    "query_id": "q1",
                    "query": "How do I reset my password?",
                    "hit": True,
                    "reciprocal_rank": 1.0,
                    "retrieved_urls": ["https://docs.example.com/password"],
                    "expected_url": "https://docs.example.com/password",
                    "latency_ms": 1.5,
                    "trace_id": "trace_001",
                },
                {
                    "query_id": "q2",
                    "query": "What are the pricing tiers?",
                    "hit": False,
                    "reciprocal_rank": 0.0,
                    "retrieved_urls": ["https://docs.example.com/other"],
                    "expected_url": "https://docs.example.com/pricing",
                    "latency_ms": 3.1,
                    "trace_id": "trace_002",
                },
            ],
            "failure_clusters": [
                {
                    "failure_type": "retrieval_failure",
                    "count": 1,
                    "common_patterns": ["pricing query mismatch"],
                },
            ],
        },
        "raptor": {
            "variant": "raptor",
            "run_name": "raptor_20260411T120000",
            "hit_rate": 0.6,
            "mrr": 0.4,
            "n_queries": 5,
            "n_hits": 3,
            "avg_latency_ms": 1.8,
            "dataset_name": "golden_eval",
            "config_snapshot": {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_provider": "minilm",
                "reranker_strategy": "passthrough",
                "retrieval_k": 5,
                "bm25_weight": 0.0,
                "vector_weight": 1.0,
            },
            "query_results": [],
            "failure_clusters": [],
        },
    },
}


@pytest.fixture
def sample_json_file(tmp_path: Path) -> Path:
    """Write sample export to a temp file."""
    p = tmp_path / "results.json"
    p.write_text(json.dumps(SAMPLE_EXPORT))
    return p


# ---------------------------------------------------------------------------
# load_from_json — basic loading
# ---------------------------------------------------------------------------


def test_load_from_json_produces_dashboard_data(sample_json_file: Path) -> None:
    """Loading from JSON returns properly typed DashboardData."""
    data = load_from_json(sample_json_file)

    assert isinstance(data, DashboardData)
    assert data.source == "local_json"
    assert not data.is_empty
    assert data.exported_at == "2026-04-11T12:00:00"


def test_load_from_json_variant_names(sample_json_file: Path) -> None:
    """Variant names are extracted correctly."""
    data = load_from_json(sample_json_file)

    assert set(data.variant_names) == {"librarian", "raptor"}


def test_load_from_json_variant_metrics(sample_json_file: Path) -> None:
    """Variant-level metrics are correctly parsed."""
    data = load_from_json(sample_json_file)

    lib = data.variants["librarian"]
    assert isinstance(lib, VariantResult)
    assert lib.hit_rate == 0.8
    assert lib.mrr == 0.65
    assert lib.n_queries == 5
    assert lib.n_hits == 4
    assert lib.avg_latency_ms == 2.3
    assert lib.dataset_name == "golden_eval"


def test_load_from_json_query_results(sample_json_file: Path) -> None:
    """Per-query results are correctly loaded."""
    data = load_from_json(sample_json_file)

    qrs = data.variants["librarian"].query_results
    assert len(qrs) == 2

    q1 = qrs[0]
    assert isinstance(q1, QueryMetric)
    assert q1.query_id == "q1"
    assert q1.hit is True
    assert q1.reciprocal_rank == 1.0
    assert q1.latency_ms == 1.5
    assert q1.trace_id == "trace_001"

    q2 = qrs[1]
    assert q2.hit is False
    assert q2.reciprocal_rank == 0.0


def test_load_from_json_failure_clusters(sample_json_file: Path) -> None:
    """Failure clusters are correctly parsed."""
    data = load_from_json(sample_json_file)

    clusters = data.variants["librarian"].failure_clusters
    assert len(clusters) == 1
    assert isinstance(clusters[0], FailureClusterData)
    assert clusters[0].failure_type == "retrieval_failure"
    assert clusters[0].count == 1
    assert "pricing query mismatch" in clusters[0].common_patterns


def test_load_from_json_config_snapshot(sample_json_file: Path) -> None:
    """Config snapshots are preserved."""
    data = load_from_json(sample_json_file)

    cs = data.variants["librarian"].config_snapshot
    assert cs["reranker_strategy"] == "cross_encoder"
    assert cs["retrieval_k"] == 10
    assert cs["bm25_weight"] == 0.3


# ---------------------------------------------------------------------------
# load_from_json — edge cases
# ---------------------------------------------------------------------------


def test_load_from_json_missing_file_raises() -> None:
    """FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_from_json("/nonexistent/results.json")


def test_load_from_json_empty_variants(tmp_path: Path) -> None:
    """Empty variants dict produces empty DashboardData."""
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"exported_at": "now", "variants": {}}))

    data = load_from_json(p)
    assert data.is_empty
    assert data.variant_names == []


def test_load_from_json_missing_query_fields(tmp_path: Path) -> None:
    """Missing optional fields in query results use defaults."""
    payload = {
        "exported_at": "now",
        "variants": {
            "test": {
                "run_name": "test_run",
                "hit_rate": 0.5,
                "mrr": 0.3,
                "n_queries": 1,
                "n_hits": 0,
                "avg_latency_ms": 1.0,
                "query_results": [
                    {
                        "query_id": "q1",
                        "query": "test query",
                        "hit": False,
                        "reciprocal_rank": 0.0,
                        # No retrieved_urls, expected_url, latency_ms, trace_id
                    }
                ],
                "failure_clusters": [],
            }
        },
    }
    p = tmp_path / "partial.json"
    p.write_text(json.dumps(payload))

    data = load_from_json(p)
    qr = data.variants["test"].query_results[0]
    assert qr.retrieved_urls == []
    assert qr.expected_url == ""
    assert qr.latency_ms == 0.0
    assert qr.trace_id == ""


# ---------------------------------------------------------------------------
# DashboardData model properties
# ---------------------------------------------------------------------------


def test_dashboard_data_is_empty_when_no_variants() -> None:
    data = DashboardData()
    assert data.is_empty


def test_dashboard_data_variant_names_sorted() -> None:
    data = DashboardData(
        variants={
            "zzz": VariantResult(
                variant_name="zzz",
                run_name="r",
                hit_rate=0,
                mrr=0,
                n_queries=0,
                n_hits=0,
                avg_latency_ms=0,
            ),
            "aaa": VariantResult(
                variant_name="aaa",
                run_name="r",
                hit_rate=0,
                mrr=0,
                n_queries=0,
                n_hits=0,
                avg_latency_ms=0,
            ),
        }
    )
    assert data.variant_names == ["aaa", "zzz"]


# ---------------------------------------------------------------------------
# load_dashboard_data — auto-loader
# ---------------------------------------------------------------------------


def test_load_dashboard_data_prefers_json(sample_json_file: Path) -> None:
    """When a JSON path is given, it takes priority over Langfuse."""
    data = load_dashboard_data(json_path=sample_json_file)
    assert data.source == "local_json"
    assert not data.is_empty


def test_load_dashboard_data_returns_empty_without_sources() -> None:
    """Without JSON or Langfuse, returns empty DashboardData."""
    data = load_dashboard_data(prefer_langfuse=False)
    assert data.source == "none"
    assert data.is_empty


def test_load_dashboard_data_missing_json_falls_through(tmp_path: Path) -> None:
    """A non-existent JSON path falls through to next source."""
    data = load_dashboard_data(
        json_path=tmp_path / "nonexistent.json",
        prefer_langfuse=False,
    )
    assert data.source == "none"


# ---------------------------------------------------------------------------
# export_results → load_from_json round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_roundtrip(tmp_path: Path) -> None:
    """Results exported by experiment CLI can be loaded by the dashboard."""
    # Run experiments
    results = await run_all_experiments(GOLDEN, CORPUS)

    # Export
    out_path = tmp_path / "results.json"
    export_results(results, out_path)
    assert out_path.exists()

    # Reload
    data = load_from_json(out_path)
    assert not data.is_empty
    assert set(data.variant_names) == set(results.keys())

    for name, result in results.items():
        loaded = data.variants[name]
        assert loaded.hit_rate == round(result.hit_rate, 3)
        assert loaded.mrr == round(result.mrr, 3)
        assert loaded.n_queries == result.n_queries
        assert loaded.n_hits == result.n_hits
        assert len(loaded.query_results) == len(result.query_results)


@pytest.mark.asyncio
async def test_exported_json_is_valid(tmp_path: Path) -> None:
    """Exported file is valid JSON with the expected top-level keys."""
    results = await run_all_experiments(GOLDEN, CORPUS)
    out_path = tmp_path / "results.json"
    export_results(results, out_path)

    raw = json.loads(out_path.read_text())
    assert "exported_at" in raw
    assert "variants" in raw
    assert isinstance(raw["variants"], dict)

    for variant_data in raw["variants"].values():
        assert "query_results" in variant_data
        assert "failure_clusters" in variant_data
        assert "config_snapshot" in variant_data


# ---------------------------------------------------------------------------
# load_from_dict — in-memory loading (fixes B4: no temp file needed)
# ---------------------------------------------------------------------------


def test_load_from_dict_produces_dashboard_data() -> None:
    """load_from_dict returns properly typed DashboardData from raw dict."""
    data = load_from_dict(SAMPLE_EXPORT)

    assert isinstance(data, DashboardData)
    assert data.source == "local_json"
    assert not data.is_empty
    assert set(data.variant_names) == {"librarian", "raptor"}


def test_load_from_dict_preserves_query_results() -> None:
    """Per-query results survive the dict → model conversion."""
    data = load_from_dict(SAMPLE_EXPORT)

    qrs = data.variants["librarian"].query_results
    assert len(qrs) == 2
    assert qrs[0].query_id == "q1"
    assert qrs[0].hit is True
    assert qrs[1].hit is False


def test_load_from_dict_empty_dict() -> None:
    """Empty dict produces empty DashboardData."""
    data = load_from_dict({})
    assert data.is_empty


def test_load_from_dict_matches_load_from_json(sample_json_file: Path) -> None:
    """load_from_dict and load_from_json produce identical results."""
    from_json = load_from_json(sample_json_file)
    from_dict = load_from_dict(SAMPLE_EXPORT)

    assert from_json.variant_names == from_dict.variant_names
    for name in from_json.variant_names:
        assert from_json.variants[name].hit_rate == from_dict.variants[name].hit_rate
        assert from_json.variants[name].mrr == from_dict.variants[name].mrr
        assert len(from_json.variants[name].query_results) == len(
            from_dict.variants[name].query_results
        )


# ---------------------------------------------------------------------------
# load_from_langfuse — graceful fallbacks (no live connection needed)
# ---------------------------------------------------------------------------


def test_load_from_langfuse_disabled_returns_empty() -> None:
    """When langfuse_enabled=False (default), returns empty DashboardData."""
    data = load_from_langfuse()
    assert data.source == "langfuse"
    assert data.is_empty


def test_load_from_langfuse_with_custom_dataset_returns_empty() -> None:
    """Custom dataset name is accepted without error when disabled."""
    data = load_from_langfuse(dataset_name="custom_dataset")
    assert data.is_empty


# ---------------------------------------------------------------------------
# load_dashboard_data — dataset_name passthrough (fixes B2)
# ---------------------------------------------------------------------------


def test_load_dashboard_data_accepts_dataset_name(sample_json_file: Path) -> None:
    """dataset_name parameter is accepted (used for Langfuse path)."""
    # JSON path takes priority, but the parameter should not raise
    data = load_dashboard_data(
        json_path=sample_json_file,
        dataset_name="custom_ds",
    )
    assert data.source == "local_json"
    assert not data.is_empty


def test_load_dashboard_data_dataset_name_without_langfuse() -> None:
    """dataset_name without Langfuse enabled falls through gracefully."""
    data = load_dashboard_data(
        prefer_langfuse=True,
        dataset_name="some_dataset",
    )
    # Langfuse is disabled in tests, so falls through to "none"
    assert data.source == "none"
