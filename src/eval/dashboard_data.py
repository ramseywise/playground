"""Data fetcher for the eval dashboard.

Loads experiment results from two sources:

1. **Langfuse API** — fetches traces and scores from the Langfuse platform for
   live dashboard views.
2. **Local JSON** — reads exported results from
   ``python -m eval.experiment run --export results.json``.

Both paths produce the same typed models so the dashboard code is source-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from librarian.config import settings

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Dashboard data models
# ---------------------------------------------------------------------------


@dataclass
class QueryMetric:
    """Per-query metrics for a single variant run."""

    query_id: str
    query: str
    hit: bool
    reciprocal_rank: float
    retrieved_urls: list[str]
    expected_url: str
    latency_ms: float
    trace_id: str = ""


@dataclass
class FailureClusterData:
    """Failure cluster summary for the dashboard."""

    failure_type: str
    count: int
    common_patterns: list[str] = field(default_factory=list)


@dataclass
class VariantResult:
    """Dashboard-ready result for a single variant."""

    variant_name: str
    run_name: str
    hit_rate: float
    mrr: float
    n_queries: int
    n_hits: int
    avg_latency_ms: float
    dataset_name: str = ""
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    query_results: list[QueryMetric] = field(default_factory=list)
    failure_clusters: list[FailureClusterData] = field(default_factory=list)


@dataclass
class DashboardData:
    """Complete dataset for the eval dashboard."""

    exported_at: str = ""
    source: str = ""  # "langfuse" or "local_json"
    variants: dict[str, VariantResult] = field(default_factory=dict)

    @property
    def variant_names(self) -> list[str]:
        return sorted(self.variants.keys())

    @property
    def is_empty(self) -> bool:
        return len(self.variants) == 0


# ---------------------------------------------------------------------------
# Shared dict → model conversion (used by JSON loader and in-memory upload)
# ---------------------------------------------------------------------------


def _parse_variant(name: str, vdata: dict[str, Any]) -> VariantResult:
    """Parse a single variant dict into a VariantResult."""
    query_results = [
        QueryMetric(
            query_id=qr["query_id"],
            query=qr["query"],
            hit=qr["hit"],
            reciprocal_rank=qr["reciprocal_rank"],
            retrieved_urls=qr.get("retrieved_urls", []),
            expected_url=qr.get("expected_url", ""),
            latency_ms=qr.get("latency_ms", 0.0),
            trace_id=qr.get("trace_id", ""),
        )
        for qr in vdata.get("query_results", [])
    ]

    failure_clusters = [
        FailureClusterData(
            failure_type=fc["failure_type"],
            count=fc["count"],
            common_patterns=fc.get("common_patterns", []),
        )
        for fc in vdata.get("failure_clusters", [])
    ]

    return VariantResult(
        variant_name=name,
        run_name=vdata.get("run_name", name),
        hit_rate=vdata.get("hit_rate", 0.0),
        mrr=vdata.get("mrr", 0.0),
        n_queries=vdata.get("n_queries", 0),
        n_hits=vdata.get("n_hits", 0),
        avg_latency_ms=vdata.get("avg_latency_ms", 0.0),
        dataset_name=vdata.get("dataset_name", ""),
        config_snapshot=vdata.get("config_snapshot", {}),
        query_results=query_results,
        failure_clusters=failure_clusters,
    )


def load_from_dict(raw: dict[str, Any]) -> DashboardData:
    """Build DashboardData from an in-memory dict (same schema as --export JSON).

    Args:
        raw: Dict with ``exported_at`` and ``variants`` keys.

    Returns:
        DashboardData with all variant results.
    """
    variants: dict[str, VariantResult] = {}
    for name, vdata in raw.get("variants", {}).items():
        variants[name] = _parse_variant(name, vdata)

    return DashboardData(
        exported_at=raw.get("exported_at", ""),
        source="local_json",
        variants=variants,
    )


# ---------------------------------------------------------------------------
# Local JSON loader
# ---------------------------------------------------------------------------


def load_from_json(path: str | Path) -> DashboardData:
    """Load experiment results from a JSON file exported by the experiment CLI.

    Args:
        path: Path to the JSON file from ``--export``.

    Returns:
        DashboardData with all variant results.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    p = Path(path)
    raw = json.loads(p.read_text())
    return load_from_dict(raw)


# ---------------------------------------------------------------------------
# Langfuse API loader — helpers
# ---------------------------------------------------------------------------


def _langfuse_init() -> Any | None:
    """Return a Langfuse client if enabled and installed, else None."""
    try:
        from langfuse import Langfuse
    except ImportError:
        log.warning("dashboard_data.langfuse.missing", msg="langfuse not installed")
        return None

    if not settings.langfuse_enabled:
        log.warning("dashboard_data.langfuse.disabled")
        return None

    try:
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception as exc:
        log.warning("dashboard_data.langfuse.init_failed", error=str(exc))
        return None


def _fetch_run_names(lf: Any, dataset_name: str) -> list[str]:
    """Fetch all run names for a dataset via the Langfuse API.

    Uses ``lf.get_dataset_runs()`` which returns DatasetRun objects.
    """
    try:
        runs = lf.get_dataset_runs(dataset_name=dataset_name)
        return sorted({r.name for r in runs})
    except Exception as exc:
        log.warning(
            "dashboard_data.langfuse.runs_fetch_failed",
            dataset=dataset_name,
            error=str(exc),
        )
        return []


def _fetch_variant_queries(lf: Any, run_name: str) -> list[QueryMetric]:
    """Fetch per-query metrics for a single run from Langfuse.

    Traces are named ``eval_{variant}_{query_id}`` with
    ``metadata.run_name == run_name`` (set by experiment.py).
    Paginates until all matching traces are collected.
    """
    variant_name = run_name.split("_")[0] if "_" in run_name else run_name
    query_results: list[QueryMetric] = []
    page_size = 50

    try:
        page = 1
        while True:
            traces = lf.fetch_traces(
                name=f"eval_{variant_name}",
                limit=page_size,
                page=page,
            )
            if not traces.data:
                break
            for trace in traces.data:
                meta = trace.metadata or {}
                if meta.get("run_name") != run_name:
                    continue
                scores = _extract_scores(lf, trace.id)
                query_results.append(_trace_to_query_metric(trace, scores))
            if len(traces.data) < page_size:
                break
            page += 1
    except Exception as exc:
        log.warning(
            "dashboard_data.langfuse.traces_fetch_failed",
            run_name=run_name,
            error=str(exc),
        )

    return query_results


def _extract_scores(lf: Any, trace_id: str) -> dict[str, float]:
    """Extract score name→value map for a trace."""
    try:
        trace_detail = lf.fetch_trace(trace_id)
        return {s.name: s.value for s in (trace_detail.scores or [])}
    except Exception:
        try:
            scores_resp = lf.fetch_scores(trace_id=trace_id)
            return {s.name: s.value for s in scores_resp.data}
        except Exception:
            return {}


def _trace_to_query_metric(trace: Any, scores: dict[str, float]) -> QueryMetric:
    """Convert a Langfuse trace + scores into a QueryMetric."""
    inp = trace.input if isinstance(trace.input, dict) else {}
    out = trace.output if isinstance(trace.output, dict) else {}
    meta = trace.metadata if isinstance(trace.metadata, dict) else {}

    # Extract query_id from trace name: "eval_{variant}_{query_id}"
    trace_name = trace.name or ""
    parts = trace_name.split("_", 2)
    query_id = parts[2] if len(parts) >= 3 else ""

    return QueryMetric(
        query_id=query_id,
        query=inp.get("query", ""),
        hit=scores.get("hit_rate", 0.0) >= 1.0,
        reciprocal_rank=scores.get("reciprocal_rank", 0.0),
        retrieved_urls=out.get("retrieved_urls", []),
        expected_url=meta.get("expected_url", ""),
        latency_ms=float(
            scores.get("retrieval_latency_ms") or meta.get("latency_ms") or 0.0
        ),
        trace_id=trace.id,
    )


def _aggregate_variant(
    run_name: str,
    query_results: list[QueryMetric],
    dataset_name: str,
) -> VariantResult:
    """Compute aggregate metrics from per-query results."""
    variant_name = run_name.split("_")[0] if "_" in run_name else run_name
    n = len(query_results)
    hits = sum(1 for qr in query_results if qr.hit)

    return VariantResult(
        variant_name=variant_name,
        run_name=run_name,
        hit_rate=hits / n if n else 0.0,
        mrr=sum(qr.reciprocal_rank for qr in query_results) / n if n else 0.0,
        n_queries=n,
        n_hits=hits,
        avg_latency_ms=sum(qr.latency_ms for qr in query_results) / n if n else 0.0,
        dataset_name=dataset_name,
        query_results=query_results,
    )


def _fetch_config_snapshots(
    lf: Any,
    variants: dict[str, VariantResult],
    *,
    limit: int = 100,
) -> None:
    """Enrich variants with config snapshots from summary traces (mutates)."""
    try:
        traces = lf.fetch_traces(name="experiment_summary", limit=limit)
        for trace in traces.data:
            meta = trace.metadata or {}
            vname = meta.get("variant", "")
            if vname in variants:
                variants[vname].config_snapshot = {
                    "embedding_model": meta.get("embedding_model", ""),
                    "reranker_strategy": meta.get("reranker_strategy", ""),
                    "bm25_weight": meta.get("bm25_weight", 0.0),
                    "vector_weight": meta.get("vector_weight", 0.0),
                    "k": meta.get("k", 0),
                }
    except Exception as exc:
        log.debug("dashboard_data.langfuse.summary_fetch_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Langfuse API loader — main entry
# ---------------------------------------------------------------------------


def load_from_langfuse(
    dataset_name: str | None = None,
    *,
    limit: int = 100,
) -> DashboardData:
    """Fetch experiment results from the Langfuse API.

    Reads dataset runs and their linked traces+scores, groups by variant.

    Args:
        dataset_name: LangFuse dataset name (defaults to settings).
        limit: Max summary traces to fetch.

    Returns:
        DashboardData populated from Langfuse.
    """
    lf = _langfuse_init()
    if lf is None:
        return DashboardData(source="langfuse")

    ds_name = dataset_name or settings.langfuse_dataset_name

    run_names = _fetch_run_names(lf, ds_name)
    if not run_names:
        log.info("dashboard_data.langfuse.no_runs", dataset=ds_name)
        return DashboardData(source="langfuse")

    variants: dict[str, VariantResult] = {}
    for run_name in run_names:
        query_results = _fetch_variant_queries(lf, run_name)
        if query_results:
            variant = _aggregate_variant(run_name, query_results, ds_name)
            variants[variant.variant_name] = variant

    _fetch_config_snapshots(lf, variants, limit=limit)

    log.info(
        "dashboard_data.langfuse.loaded",
        dataset=ds_name,
        n_variants=len(variants),
        n_runs=len(run_names),
    )

    return DashboardData(
        exported_at=datetime.now().isoformat(),
        source="langfuse",
        variants=variants,
    )


# ---------------------------------------------------------------------------
# Auto-loader: tries Langfuse first, falls back to local JSON
# ---------------------------------------------------------------------------


def load_dashboard_data(
    json_path: str | Path | None = None,
    *,
    prefer_langfuse: bool = True,
    dataset_name: str | None = None,
) -> DashboardData:
    """Load dashboard data from the best available source.

    Priority:
    1. If ``json_path`` is given, load from local JSON.
    2. If ``prefer_langfuse`` and Langfuse is configured, fetch from API.
    3. Return empty DashboardData.

    Args:
        json_path: Optional path to an exported JSON file.
        prefer_langfuse: Whether to try Langfuse when no JSON path is given.
        dataset_name: LangFuse dataset name override.

    Returns:
        DashboardData from the first successful source.
    """
    # Explicit JSON file takes precedence
    if json_path is not None:
        p = Path(json_path)
        if p.exists():
            log.info("dashboard_data.load.json", path=str(p))
            return load_from_json(p)
        log.warning("dashboard_data.load.json_not_found", path=str(p))

    # Try Langfuse
    if prefer_langfuse and settings.langfuse_enabled:
        log.info("dashboard_data.load.langfuse")
        data = load_from_langfuse(dataset_name=dataset_name)
        if not data.is_empty:
            return data
        log.info("dashboard_data.load.langfuse_empty")

    return DashboardData(source="none")
