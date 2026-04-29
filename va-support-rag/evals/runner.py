"""Experiment runner: run retrieval variants against golden datasets, log to LangFuse.

Usage:
    uv run python -m evals.runner upload
    uv run python -m evals.runner run
    uv run python -m evals.runner run --variant rag-poc
    uv run python -m evals.runner run --export results.json
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel

from evals.utils.loaders import load_golden_from_faq_csv, load_golden_from_jsonl
from evals.utils.models import (
    ExperimentResult,
    FailureClusterSummary,
    GoldenSample,
    QueryResult,
)
from evals.utils.settings import settings
from evals.utils.tracing import FailureClusterer, PipelineTracer

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------


class VariantSettings(BaseModel):
    """Named pipeline parameters for a retrieval variant."""

    embedding_provider: str = "multilingual"
    embedding_model: str = "intfloat/multilingual-e5-large"
    retrieval_strategy: str = "rag_poc_local"
    reranker_strategy: str = "cross_encoder"
    retrieval_k: int = 10
    reranker_top_k: int = 3
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    confidence_threshold: float = 0.0
    max_crag_retries: int = 0
    anthropic_api_key: str = "test"
    google_project_id: str = ""
    google_datastore_id: str = ""
    bedrock_knowledge_base_id: str = ""
    bedrock_model_arn: str = ""
    bedrock_region: str = ""
    model_gemini: str = ""
    gemini_api_key: str = ""

    model_config = {"extra": "ignore"}


VARIANTS: dict[str, VariantSettings] = {
    "rag-poc": VariantSettings(
        retrieval_strategy="rag_poc_local",
        reranker_strategy="ensemble",
        retrieval_k=12,
        reranker_top_k=5,
    ),
    "bedrock-live": VariantSettings(
        embedding_provider="aws_titan",
        retrieval_strategy="bedrock",
        reranker_strategy="passthrough",
        retrieval_k=5,
        reranker_top_k=5,
        bm25_weight=0.0,
        vector_weight=1.0,
    ),
    "google-adk": VariantSettings(
        embedding_provider="google",
        retrieval_strategy="google_adk",
        reranker_strategy="passthrough",
        retrieval_k=5,
        reranker_top_k=5,
        bm25_weight=0.0,
        vector_weight=1.0,
    ),
}


# ---------------------------------------------------------------------------
# LangFuse helpers — graceful no-op when unconfigured
# ---------------------------------------------------------------------------


def _get_langfuse_client() -> Any | None:
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except ImportError:
        log.warning("experiment.langfuse.missing", msg="langfuse not installed")
        return None
    except Exception as exc:
        log.warning("experiment.langfuse.init_failed", error=str(exc))
        return None


def _langfuse_create_dataset(lf: Any, name: str) -> None:
    try:
        lf.create_dataset(name=name)
    except Exception as exc:
        log.warning("experiment.langfuse.create_dataset_failed", error=str(exc))


def _langfuse_create_item(lf: Any, dataset_name: str, sample: GoldenSample) -> None:
    try:
        lf.create_dataset_item(
            dataset_name=dataset_name,
            input={"query": sample.query},
            expected_output={"doc_url": sample.expected_doc_url},
            metadata={
                "query_id": sample.query_id,
                "category": sample.category,
                "difficulty": sample.difficulty,
                "language": sample.language,
                "validation_level": sample.validation_level,
                "relevant_chunk_ids": sample.relevant_chunk_ids,
            },
            id=sample.query_id,
        )
    except Exception as exc:
        log.warning(
            "experiment.langfuse.create_item_failed",
            query_id=sample.query_id,
            error=str(exc),
        )


def _langfuse_log_trace(
    lf: Any, variant_name: str, qr: QueryResult, run_name: str, dataset_name: str
) -> str:
    try:
        trace = lf.trace(
            name=f"eval_{variant_name}_{qr.query_id}",
            input={"query": qr.query},
            output={"hit": qr.hit, "retrieved_urls": qr.retrieved_urls},
            metadata={
                "variant": variant_name,
                "run_name": run_name,
                "expected_url": qr.expected_url,
                "latency_ms": qr.latency_ms,
            },
        )
        lf.score(trace_id=trace.id, name="hit_rate", value=1.0 if qr.hit else 0.0)
        lf.score(trace_id=trace.id, name="reciprocal_rank", value=qr.reciprocal_rank)
        lf.score(trace_id=trace.id, name="retrieval_latency_ms", value=qr.latency_ms)
        try:
            dataset = lf.get_dataset(dataset_name)
            for item in dataset.items:
                if item.id == qr.query_id:
                    item.link(trace, run_name=run_name)
                    break
        except Exception:
            pass
        return trace.id
    except Exception as exc:
        log.warning(
            "experiment.langfuse.trace_failed", query_id=qr.query_id, error=str(exc)
        )
        return ""


def _langfuse_flush(lf: Any) -> None:
    try:
        lf.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Dataset upload
# ---------------------------------------------------------------------------


def upload_golden_dataset(
    samples: list[GoldenSample],
    dataset_name: str | None = None,
    *,
    langfuse_client: Any | None = None,
) -> int:
    """Upload golden samples to LangFuse as a named dataset. Returns items uploaded."""
    lf = langfuse_client or _get_langfuse_client()
    if lf is None:
        log.warning("experiment.upload.skipped", reason="langfuse not available")
        return 0
    name = dataset_name or settings.langfuse_dataset_name
    _langfuse_create_dataset(lf, name)
    for sample in samples:
        _langfuse_create_item(lf, name, sample)
    _langfuse_flush(lf)
    log.info("experiment.upload.done", dataset=name, n_items=len(samples))
    return len(samples)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _score_hit(expected_url: str, urls: list[str]) -> tuple[bool, float]:
    hit = expected_url in urls
    rr = next((1.0 / (i + 1) for i, u in enumerate(urls) if u == expected_url), 0.0)
    return hit, rr


def _score_doc_level(expected: str, graded: list[Any]) -> tuple[bool, float]:
    if not expected:
        return False, 0.0
    for i, gc in enumerate(graded):
        m = gc.chunk.metadata
        pool = {x for x in (m.url, m.doc_id, gc.chunk.id) if x}
        if (
            expected in pool
            or (m.url and expected in m.url)
            or (m.doc_id and expected in m.doc_id)
        ):
            return True, 1.0 / (i + 1)
    return False, 0.0


def _score_chunk_level(
    expected_ids: list[str], graded: list[Any]
) -> tuple[bool | None, float | None]:
    if not expected_ids:
        return None, None
    want = set(expected_ids)
    best_rr = max(
        (1.0 / (i + 1) for i, gc in enumerate(graded) if gc.chunk.id in want),
        default=0.0,
    )
    return (best_rr > 0.0), (best_rr or 0.0)


def _aggregate_results(
    query_results: list[QueryResult],
    *,
    variant_name: str,
    ds_name: str,
    run_name: str,
    config_snapshot: dict[str, Any],
    lf: Any | None,
) -> ExperimentResult:
    n = len(query_results)
    hits = sum(1 for qr in query_results if qr.hit)
    hit_rate = hits / n if n else 0.0
    mrr = sum(qr.reciprocal_rank for qr in query_results) / n if n else 0.0
    avg_latency = sum(qr.latency_ms for qr in query_results) / n if n else 0.0

    chunk_scored = [qr for qr in query_results if qr.chunk_hit is not None]
    n_chunk = len(chunk_scored)
    chunk_hit_rate = (
        sum(1 for qr in chunk_scored if qr.chunk_hit) / n_chunk if n_chunk else None
    )
    chunk_mrr_val = (
        sum(qr.chunk_reciprocal_rank or 0.0 for qr in chunk_scored) / n_chunk
        if n_chunk
        else None
    )

    tracer = PipelineTracer()
    for qr in query_results:
        trace = tracer.create_trace(qr.query_id, qr.query)
        trace.status = "success" if qr.hit else "failure"
        trace.confidence = qr.reciprocal_rank
        if not qr.hit:
            trace.failure_reason = "expected_doc_not_in_top_k"
    clusters = FailureClusterer().cluster_failures(tracer.get_failure_traces())

    if lf is not None:
        try:
            summary = lf.trace(
                name=f"experiment_summary_{variant_name}",
                metadata={
                    "variant": variant_name,
                    "run_name": run_name,
                    **config_snapshot,
                },
            )
            lf.score(trace_id=summary.id, name="hit_rate_at_k", value=hit_rate)
            lf.score(trace_id=summary.id, name="mrr", value=mrr)
            lf.score(trace_id=summary.id, name="avg_latency_ms", value=avg_latency)
        except Exception as exc:
            log.warning("experiment.langfuse.summary_failed", error=str(exc))
        _langfuse_flush(lf)

    result = ExperimentResult(
        variant_name=variant_name,
        dataset_name=ds_name,
        run_name=run_name,
        hit_rate=hit_rate,
        mrr=mrr,
        n_queries=n,
        n_hits=hits,
        avg_latency_ms=avg_latency,
        query_results=query_results,
        failure_clusters=[
            FailureClusterSummary(
                failure_type=c.failure_type,
                count=c.count,
                common_patterns=c.common_patterns,
            )
            for c in clusters
        ],
        config_snapshot=config_snapshot,
        chunk_hit_rate=chunk_hit_rate,
        chunk_mrr=chunk_mrr_val,
        n_chunk_queries=n_chunk,
    )
    log.info(
        "experiment.variant.done",
        variant=variant_name,
        hit_rate=hit_rate,
        mrr=mrr,
        n=n,
        avg_latency_ms=round(avg_latency, 1),
    )
    return result


async def _run_query_loop(
    variant_name: str,
    samples: list[GoldenSample],
    fetch_fn: Any,
    lf: Any | None,
    run_name: str,
    ds_name: str,
) -> list[QueryResult]:
    query_results: list[QueryResult] = []
    for sample in samples:
        try:
            urls, latency_ms, answer = await fetch_fn(sample)
            hit, rr = _score_hit(sample.expected_doc_url, urls)
            qr = QueryResult(
                query_id=sample.query_id,
                query=sample.query,
                hit=hit,
                reciprocal_rank=rr,
                retrieved_urls=urls[:5],
                expected_url=sample.expected_doc_url,
                latency_ms=latency_ms,
                answer=answer,
            )
        except Exception as exc:
            log.warning(
                f"experiment.{variant_name}.query_failed",
                query_id=sample.query_id,
                error=str(exc),
            )
            qr = QueryResult(
                query_id=sample.query_id,
                query=sample.query,
                hit=False,
                reciprocal_rank=0.0,
                retrieved_urls=[],
                expected_url=sample.expected_doc_url,
                latency_ms=0.0,
            )
        if lf is not None:
            qr.trace_id = _langfuse_log_trace(lf, variant_name, qr, run_name, ds_name)
        query_results.append(qr)
    return query_results


# ---------------------------------------------------------------------------
# Variant runners
# ---------------------------------------------------------------------------


async def _run_rag_poc_local(
    variant_name: str,
    samples: list[GoldenSample],
    *,
    cfg: VariantSettings,
    run_name: str,
    ds_name: str,
    lf: Any | None,
) -> ExperimentResult:
    from core.config import RAG_RETRIEVAL_QUERY_TRANSFORM
    from core.observability import configure_runtime
    from orchestrator.langgraph.nodes.retriever import expand_queries_for_retrieval
    from rag.retrieval.pipeline import get_ensemble_retriever

    configure_runtime()
    ensemble = get_ensemble_retriever()
    k = max(1, cfg.retrieval_k)
    query_results: list[QueryResult] = []

    for sample in samples:
        t0 = time.perf_counter()
        try:
            retrieval_queries, _ = expand_queries_for_retrieval(sample.query)
            graded = await ensemble.retrieve(retrieval_queries, k=k)
            latency_ms = (time.perf_counter() - t0) * 1000
            urls = [
                gc.chunk.metadata.url or gc.chunk.metadata.doc_id or "" for gc in graded
            ]
            hit, rr = (
                _score_doc_level(sample.expected_doc_url, graded)
                if sample.expected_doc_url
                else (False, 0.0)
            )
            ch_hit, ch_rr = _score_chunk_level(sample.relevant_chunk_ids, graded)
            qr = QueryResult(
                query_id=sample.query_id,
                query=sample.query,
                hit=hit,
                reciprocal_rank=rr,
                retrieved_urls=urls[:5],
                expected_url=sample.expected_doc_url,
                latency_ms=latency_ms,
                retrieved_chunk_ids=[gc.chunk.id for gc in graded][:k],
                chunk_hit=ch_hit,
                chunk_reciprocal_rank=ch_rr,
            )
        except Exception as exc:
            log.warning(
                "experiment.rag_poc_local.query_failed",
                query_id=sample.query_id,
                error=str(exc),
            )
            qr = QueryResult(
                query_id=sample.query_id,
                query=sample.query,
                hit=False,
                reciprocal_rank=0.0,
                retrieved_urls=[],
                expected_url=sample.expected_doc_url,
                latency_ms=0.0,
                retrieved_chunk_ids=[],
            )
        if lf is not None:
            qr.trace_id = _langfuse_log_trace(lf, variant_name, qr, run_name, ds_name)
        query_results.append(qr)

    return _aggregate_results(
        query_results,
        variant_name=variant_name,
        ds_name=ds_name,
        run_name=run_name,
        lf=lf,
        config_snapshot={
            "retrieval_strategy": "rag_poc_local",
            "retrieval_k": k,
            "rag_retrieval_query_transform": RAG_RETRIEVAL_QUERY_TRANSFORM,
        },
    )


async def _run_bedrock(
    variant_name: str,
    samples: list[GoldenSample],
    *,
    cfg: VariantSettings,
    run_name: str,
    ds_name: str,
    lf: Any | None,
) -> ExperimentResult:
    from clients.bedrock_KB import BedrockKBClient

    client = BedrockKBClient(cfg)

    async def fetch(sample: GoldenSample) -> tuple[list[str], float, str | None]:
        t0 = time.perf_counter()
        resp = await client.aquery(sample.query)
        return (
            [c["url"] for c in resp.citations],
            (time.perf_counter() - t0) * 1000,
            resp.response,
        )

    query_results = await _run_query_loop(
        variant_name, samples, fetch, lf, run_name, ds_name
    )
    return _aggregate_results(
        query_results,
        variant_name=variant_name,
        ds_name=ds_name,
        run_name=run_name,
        lf=lf,
        config_snapshot={
            "retrieval_strategy": "bedrock",
            "kb_id": cfg.bedrock_knowledge_base_id,
            "model_arn": cfg.bedrock_model_arn,
            "retrieval_k": cfg.retrieval_k,
        },
    )


async def _run_google_adk(
    variant_name: str,
    samples: list[GoldenSample],
    *,
    cfg: VariantSettings,
    run_name: str,
    ds_name: str,
    lf: Any | None,
) -> ExperimentResult:
    from clients.google_vertex import GoogleRAGClient

    client = GoogleRAGClient(cfg)

    async def fetch(sample: GoldenSample) -> tuple[list[str], float, str | None]:
        t0 = time.perf_counter()
        resp = await client.aquery(sample.query)
        return (
            [c["url"] for c in resp.citations],
            (time.perf_counter() - t0) * 1000,
            resp.response,
        )

    query_results = await _run_query_loop(
        variant_name, samples, fetch, lf, run_name, ds_name
    )
    return _aggregate_results(
        query_results,
        variant_name=variant_name,
        ds_name=ds_name,
        run_name=run_name,
        lf=lf,
        config_snapshot={
            "retrieval_strategy": "google_adk",
            "google_project_id": cfg.google_project_id,
            "google_datastore_id": cfg.google_datastore_id,
            "retrieval_k": cfg.retrieval_k,
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_variant_experiment(
    variant_name: str,
    golden_samples: list[GoldenSample],
    *,
    cfg: VariantSettings | None = None,
    dataset_name: str | None = None,
    langfuse_client: Any | None = None,
) -> ExperimentResult:
    """Run a single retrieval variant against golden samples."""
    cfg = cfg or VARIANTS[variant_name]
    ds_name = dataset_name or settings.langfuse_dataset_name
    run_name = f"{variant_name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    lf = langfuse_client or _get_langfuse_client()

    dispatch = {
        "rag_poc_local": _run_rag_poc_local,
        "bedrock": _run_bedrock,
        "google_adk": _run_google_adk,
    }
    runner = dispatch.get(cfg.retrieval_strategy)
    if runner is None:
        raise ValueError(f"Unsupported retrieval_strategy: {cfg.retrieval_strategy!r}")
    return await runner(
        variant_name, golden_samples, cfg=cfg, run_name=run_name, ds_name=ds_name, lf=lf
    )


async def run_all_experiments(
    golden_samples: list[GoldenSample],
    *,
    variants: dict[str, VariantSettings] | None = None,
    dataset_name: str | None = None,
    langfuse_client: Any | None = None,
) -> dict[str, ExperimentResult]:
    """Run all configured variants and return comparison results."""
    variant_configs = variants or VARIANTS
    results: dict[str, ExperimentResult] = {}
    for name, cfg in variant_configs.items():
        if cfg.retrieval_strategy == "bedrock" and not cfg.bedrock_knowledge_base_id:
            log.info(
                "experiment.variant.skipped",
                variant=name,
                reason="BEDROCK_KNOWLEDGE_BASE_ID not set",
            )
            continue
        if cfg.retrieval_strategy == "google_adk" and not (
            cfg.google_datastore_id or cfg.google_project_id
        ):
            log.info(
                "experiment.variant.skipped",
                variant=name,
                reason="GOOGLE_DATASTORE_ID not set",
            )
            continue
        results[name] = await run_variant_experiment(
            name,
            golden_samples,
            cfg=cfg,
            dataset_name=dataset_name,
            langfuse_client=langfuse_client,
        )
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def print_comparison_table(results: dict[str, ExperimentResult]) -> None:
    if not results:
        print(
            "\n  No variants ran — all were skipped. See logs above for skip reasons."
        )  # noqa: T201
        return
    sep = "-" * 80
    header = f"\n{'Variant':<12} {'hit_rate':>10} {'MRR':>10} {'ch_hit':>10} {'ch_MRR':>10} {'n':>5} {'hits':>5} {'avg_ms':>8} {'failures'}"
    print(f"\n{sep}\n  Experiment Comparison\n{sep}")  # noqa: T201
    print(header)  # noqa: T201
    print(sep)  # noqa: T201
    for name, r in results.items():
        failure_str = (
            ", ".join(f"{c.failure_type}×{c.count}" for c in r.failure_clusters)
            or "none"
        )
        ch_h = f"{r.chunk_hit_rate:.3f}" if r.chunk_hit_rate is not None else "   —"
        ch_m = f"{r.chunk_mrr:.3f}" if r.chunk_mrr is not None else "   —"
        print(
            f"  {name:<10} {r.hit_rate:>10.3f} {r.mrr:>10.3f} {ch_h:>10} {ch_m:>10} {r.n_queries:>5} {r.n_hits:>5} {r.avg_latency_ms:>7.1f} [{failure_str}]"
        )  # noqa: T201
    print(sep)  # noqa: T201
    if settings.langfuse_enabled:
        print(f"\n  LangFuse: traces logged → {settings.langfuse_host}")  # noqa: T201
    else:
        print("\n  LangFuse: disabled — set LANGFUSE_ENABLED=true to log traces")  # noqa: T201
    print()  # noqa: T201


def export_results(
    results: dict[str, ExperimentResult], output_path: str | Path
) -> Path:
    """Export experiment results to JSON."""
    path = Path(output_path)
    payload: dict[str, Any] = {
        "exported_at": datetime.now().isoformat(),
        "variants": {
            name: {
                **result.summary_dict(),
                "dataset_name": result.dataset_name,
                "query_results": [
                    {
                        "query_id": qr.query_id,
                        "query": qr.query,
                        "hit": qr.hit,
                        "reciprocal_rank": qr.reciprocal_rank,
                        "retrieved_urls": qr.retrieved_urls,
                        "expected_url": qr.expected_url,
                        "latency_ms": qr.latency_ms,
                        "trace_id": qr.trace_id,
                        "retrieved_chunk_ids": qr.retrieved_chunk_ids,
                        "chunk_hit": qr.chunk_hit,
                        "chunk_reciprocal_rank": qr.chunk_reciprocal_rank,
                    }
                    for qr in result.query_results
                ],
                "failure_clusters": [
                    {
                        "failure_type": c.failure_type,
                        "count": c.count,
                        "common_patterns": c.common_patterns,
                    }
                    for c in result.failure_clusters
                ],
                "config_snapshot": result.config_snapshot,
            }
            for name, result in results.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("experiment.export.done", path=str(path), n_variants=len(results))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_samples(path: str | None = None) -> list[GoldenSample]:
    dataset_path = path or settings.eval_dataset_path
    if not dataset_path:
        log.info(
            "experiment.load.fallback",
            msg="No EVAL_DATASET_PATH — using placeholder sample",
        )
        return [
            GoldenSample(
                query_id="q1",
                query="placeholder",
                expected_doc_url="https://example.com/doc",
                category="smoke",
                language="en",
                difficulty="easy",
                validation_level="synthetic",
            )
        ]
    p = Path(dataset_path)
    return (
        load_golden_from_faq_csv(p)
        if p.suffix.lower() == ".csv"
        else load_golden_from_jsonl(dataset_path)
    )


def _cli_upload(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Upload golden dataset to LangFuse")
    parser.add_argument(
        "--path", help="Path to golden JSONL (overrides EVAL_DATASET_PATH)"
    )
    parser.add_argument("--dataset", help="LangFuse dataset name", default=None)
    parsed = parser.parse_args(args)
    n = upload_golden_dataset(_load_samples(parsed.path), dataset_name=parsed.dataset)
    print(
        f"Uploaded {n} samples to LangFuse"
        if n
        else "Upload skipped — check LANGFUSE_ENABLED and credentials"
    )  # noqa: T201


def _cli_run(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run retrieval experiments")
    parser.add_argument(
        "--path", help="Path to golden JSONL (overrides EVAL_DATASET_PATH)"
    )
    parser.add_argument(
        "--variant", help="Single variant to run (default: all)", default=None
    )
    parser.add_argument("--dataset", help="LangFuse dataset name", default=None)
    parser.add_argument(
        "--export", help="Export results to JSON file", default=None, metavar="FILE"
    )
    parsed = parser.parse_args(args)

    samples = _load_samples(parsed.path)
    dataset_name = parsed.dataset or settings.langfuse_dataset_name

    if parsed.variant:
        if parsed.variant not in VARIANTS:
            print(
                f"Unknown variant '{parsed.variant}'. Available: {list(VARIANTS.keys())}"
            )  # noqa: T201
            sys.exit(1)
        all_results = {
            parsed.variant: asyncio.run(
                run_variant_experiment(
                    parsed.variant, samples, dataset_name=dataset_name
                )
            )
        }
    else:
        all_results = asyncio.run(
            run_all_experiments(samples, dataset_name=dataset_name)
        )

    print_comparison_table(all_results)
    if parsed.export:
        print(f"Results exported to {export_results(all_results, parsed.export)}")  # noqa: T201


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python -m evals.runner <upload|run> [options]")  # noqa: T201
        print("  upload  Upload golden dataset to LangFuse")  # noqa: T201
        print("  run     Run variant experiments (all or single --variant NAME)")  # noqa: T201
        sys.exit(0)
    command = sys.argv[1]
    if command == "upload":
        _cli_upload(sys.argv[2:])
    elif command == "run":
        _cli_run(sys.argv[2:])
    else:
        print(f"Unknown command '{command}'. Use 'upload' or 'run'.")  # noqa: T201
        sys.exit(1)


if __name__ == "__main__":
    main()
