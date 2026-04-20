"""LangFuse experiment runner for variant comparison.

Uploads golden datasets to LangFuse, runs retrieval variants against them,
scores results (hit_rate@k, MRR, per-grader), and links traces for
dashboard comparison.

When LANGFUSE_ENABLED=false (or langfuse not installed), experiments still
run and print results — LangFuse logging is a no-op.

Usage:
    # Upload golden dataset to LangFuse
    uv run python -m eval.experiment upload

    # Run all three variants (librarian, raptor, bedrock)
    uv run python -m eval.experiment run

    # Run a single variant
    uv run python -m eval.experiment run --variant librarian

    # Custom dataset path (overrides EVAL_DATASET_PATH)
    uv run python -m eval.experiment upload --path /path/to/golden.jsonl
    uv run python -m eval.experiment run --path /path/to/golden.jsonl

    # Export results as JSON for the eval dashboard
    uv run python -m eval.experiment run --export results.json
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

from eval.loaders import load_golden_from_jsonl
from eval.models import ExperimentResult, FailureClusterSummary, QueryResult
from eval.variants import VARIANTS
from librarian.config import LibrarySettings, settings
from librarian.schemas.chunks import Chunk, ChunkMetadata
from librarian.schemas.retrieval import RetrievalResult
from librarian.ingestion.tasks.models import GoldenSample, RetrievalMetrics
from librarian.ingestion.tasks.tracing import FailureCluster, FailureClusterer

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# LangFuse helpers — graceful no-op when unconfigured
# ---------------------------------------------------------------------------


def _get_langfuse_client() -> Any | None:
    """Return a Langfuse client if enabled and installed, else None."""
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


def _langfuse_create_dataset(lf: Any, name: str) -> Any | None:
    """Create a LangFuse dataset (idempotent)."""
    try:
        return lf.create_dataset(name=name)
    except Exception as exc:
        log.warning("experiment.langfuse.create_dataset_failed", error=str(exc))
        return None


def _langfuse_create_item(
    lf: Any,
    dataset_name: str,
    sample: GoldenSample,
) -> None:
    """Create a single dataset item in LangFuse."""
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
    lf: Any,
    variant_name: str,
    qr: QueryResult,
    run_name: str,
    dataset_name: str,
) -> str:
    """Create a LangFuse trace for a single query evaluation and link to dataset item."""
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
        # Score the trace
        lf.score(
            trace_id=trace.id,
            name="hit_rate",
            value=1.0 if qr.hit else 0.0,
        )
        lf.score(
            trace_id=trace.id,
            name="reciprocal_rank",
            value=qr.reciprocal_rank,
        )
        lf.score(
            trace_id=trace.id,
            name="retrieval_latency_ms",
            value=qr.latency_ms,
        )
        # Link to dataset item
        try:
            dataset = lf.get_dataset(dataset_name)
            for item in dataset.items:
                if item.id == qr.query_id:
                    item.link(trace, run_name=run_name)
                    break
        except Exception:
            pass  # linking is best-effort

        return trace.id
    except Exception as exc:
        log.warning(
            "experiment.langfuse.trace_failed",
            query_id=qr.query_id,
            error=str(exc),
        )
        return ""


def _langfuse_flush(lf: Any) -> None:
    """Flush pending LangFuse events."""
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
    """Upload golden samples to LangFuse as a named dataset.

    Returns the number of items uploaded. Returns 0 if LangFuse is unavailable.
    """
    lf = langfuse_client or _get_langfuse_client()
    if lf is None:
        log.warning("experiment.upload.skipped", reason="langfuse not available")
        return 0

    name = dataset_name or settings.langfuse_dataset_name
    _langfuse_create_dataset(lf, name)

    uploaded = 0
    for sample in samples:
        _langfuse_create_item(lf, name, sample)
        uploaded += 1

    _langfuse_flush(lf)
    log.info(
        "experiment.upload.done",
        dataset=name,
        n_items=uploaded,
    )
    return uploaded


# ---------------------------------------------------------------------------
# Shared eval helpers
# ---------------------------------------------------------------------------


# Re-export from canonical location for backward compat
from orchestration.google_adk.context import (  # noqa: E402
    build_adk_context as _build_adk_context,
    extract_urls_from_adk_events as _extract_urls_from_adk_events,
)


def _aggregate_results(
    query_results: list[QueryResult],
    *,
    variant_name: str,
    ds_name: str,
    run_name: str,
    config_snapshot: dict[str, Any],
    lf: Any | None,
) -> ExperimentResult:
    """Shared aggregation logic for all experiment variants.

    Computes hit_rate, MRR, avg_latency, clusters failures, logs to
    LangFuse, and returns an ExperimentResult. Eliminates the ~60 lines
    of duplicated code across 5+ experiment runners.
    """
    n = len(query_results)
    hits = sum(1 for qr in query_results if qr.hit)
    hit_rate = hits / n if n else 0.0
    mrr = sum(qr.reciprocal_rank for qr in query_results) / n if n else 0.0
    avg_latency = sum(qr.latency_ms for qr in query_results) / n if n else 0.0

    # Cluster failures
    from librarian.ingestion.tasks.tracing import PipelineTracer

    clusterer = FailureClusterer()
    tracer = PipelineTracer()
    for qr in query_results:
        trace = tracer.create_trace(qr.query_id, qr.query)
        trace.status = "success" if qr.hit else "failure"
        trace.confidence = qr.reciprocal_rank
        if not qr.hit:
            trace.failure_reason = "expected_doc_not_in_top_k"
    clusters = clusterer.cluster_failures(tracer.get_failure_traces())

    # LangFuse summary
    if lf is not None:
        try:
            summary_trace = lf.trace(
                name=f"experiment_summary_{variant_name}",
                metadata={
                    "variant": variant_name,
                    "run_name": run_name,
                    **config_snapshot,
                },
            )
            lf.score(trace_id=summary_trace.id, name="hit_rate_at_k", value=hit_rate)
            lf.score(trace_id=summary_trace.id, name="mrr", value=mrr)
            lf.score(
                trace_id=summary_trace.id, name="avg_latency_ms", value=avg_latency
            )
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


def _score_hit(expected_url: str, urls: list[str]) -> tuple[bool, float]:
    """Check if expected_url is in retrieved urls, compute reciprocal rank."""
    hit = expected_url in urls
    rr = next(
        (1.0 / (i + 1) for i, u in enumerate(urls) if u == expected_url),
        0.0,
    )
    return hit, rr


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


async def _run_bedrock_experiment(
    variant_name: str,
    golden_samples: list[GoldenSample],
    *,
    cfg: LibrarySettings,
    run_name: str,
    ds_name: str,
    lf: Any | None,
) -> ExperimentResult:
    """Run experiment via real Bedrock KB API (RetrieveAndGenerate)."""
    from clients.bedrock_KB import BedrockKBClient

    client = BedrockKBClient(cfg)
    clusterer = FailureClusterer()
    query_results: list[QueryResult] = []

    for sample in golden_samples:
        try:
            t0 = time.perf_counter()
            resp = await client.aquery(sample.query)
            latency_ms = (time.perf_counter() - t0) * 1000

            urls = [c["url"] for c in resp.citations]
            hit = sample.expected_doc_url in urls
            rr = next(
                (
                    1.0 / (i + 1)
                    for i, u in enumerate(urls)
                    if u == sample.expected_doc_url
                ),
                0.0,
            )

            qr = QueryResult(
                query_id=sample.query_id,
                query=sample.query,
                hit=hit,
                reciprocal_rank=rr,
                retrieved_urls=urls[:5],
                expected_url=sample.expected_doc_url,
                latency_ms=latency_ms,
                answer=resp.response,
            )
        except Exception as exc:
            log.warning(
                "experiment.bedrock.query_failed",
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

    # Aggregate
    n = len(query_results)
    hits = sum(1 for qr in query_results if qr.hit)
    hit_rate = hits / n if n else 0.0
    mrr = sum(qr.reciprocal_rank for qr in query_results) / n if n else 0.0
    avg_latency = sum(qr.latency_ms for qr in query_results) / n if n else 0.0

    # Cluster failures
    from librarian.ingestion.tasks.tracing import PipelineTracer

    tracer = PipelineTracer()
    for qr in query_results:
        trace = tracer.create_trace(qr.query_id, qr.query)
        trace.status = "success" if qr.hit else "failure"
        trace.confidence = qr.reciprocal_rank
        if not qr.hit:
            trace.failure_reason = "expected_doc_not_in_top_k"
    clusters = clusterer.cluster_failures(tracer.get_failure_traces())

    if lf is not None:
        try:
            summary_trace = lf.trace(
                name=f"experiment_summary_{variant_name}",
                metadata={
                    "variant": variant_name,
                    "run_name": run_name,
                    "kb_id": cfg.bedrock_knowledge_base_id,
                    "model_arn": cfg.bedrock_model_arn,
                },
            )
            lf.score(trace_id=summary_trace.id, name="hit_rate_at_k", value=hit_rate)
            lf.score(trace_id=summary_trace.id, name="mrr", value=mrr)
            lf.score(
                trace_id=summary_trace.id, name="avg_latency_ms", value=avg_latency
            )
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
        config_snapshot={
            "retrieval_strategy": "bedrock",
            "kb_id": cfg.bedrock_knowledge_base_id,
            "model_arn": cfg.bedrock_model_arn,
            "region": cfg.bedrock_region or cfg.s3_region or "default",
            "retrieval_k": cfg.retrieval_k,
            "embedding_provider": "aws_titan",
            "reranker_strategy": "n/a",
            "bm25_weight": "n/a",
            "vector_weight": "n/a",
        },
    )

    log.info(
        "experiment.bedrock.done",
        variant=variant_name,
        hit_rate=hit_rate,
        mrr=mrr,
        n=n,
        avg_latency_ms=round(avg_latency, 1),
    )
    return result


async def _run_google_adk_experiment(
    variant_name: str,
    golden_samples: list[GoldenSample],
    *,
    cfg: LibrarySettings,
    run_name: str,
    ds_name: str,
    lf: Any | None,
) -> ExperimentResult:
    """Run experiment via real Google Gemini + Vertex AI Search grounding."""
    from clients.google_vertex import GoogleRAGClient

    client = GoogleRAGClient(cfg)
    clusterer = FailureClusterer()
    query_results: list[QueryResult] = []

    for sample in golden_samples:
        try:
            t0 = time.perf_counter()
            resp = await client.aquery(sample.query)
            latency_ms = (time.perf_counter() - t0) * 1000

            urls = [c["url"] for c in resp.citations]
            hit = sample.expected_doc_url in urls
            rr = next(
                (
                    1.0 / (i + 1)
                    for i, u in enumerate(urls)
                    if u == sample.expected_doc_url
                ),
                0.0,
            )

            qr = QueryResult(
                query_id=sample.query_id,
                query=sample.query,
                hit=hit,
                reciprocal_rank=rr,
                retrieved_urls=urls[:5],
                expected_url=sample.expected_doc_url,
                latency_ms=latency_ms,
                answer=resp.response,
            )
        except Exception as exc:
            log.warning(
                "experiment.google_adk.query_failed",
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

    # Aggregate
    n = len(query_results)
    hits = sum(1 for qr in query_results if qr.hit)
    hit_rate = hits / n if n else 0.0
    mrr = sum(qr.reciprocal_rank for qr in query_results) / n if n else 0.0
    avg_latency = sum(qr.latency_ms for qr in query_results) / n if n else 0.0

    # Cluster failures
    from librarian.ingestion.tasks.tracing import PipelineTracer

    tracer = PipelineTracer()
    for qr in query_results:
        trace = tracer.create_trace(qr.query_id, qr.query)
        trace.status = "success" if qr.hit else "failure"
        trace.confidence = qr.reciprocal_rank
        if not qr.hit:
            trace.failure_reason = "expected_doc_not_in_top_k"
    clusters = clusterer.cluster_failures(tracer.get_failure_traces())

    if lf is not None:
        try:
            summary_trace = lf.trace(
                name=f"experiment_summary_{variant_name}",
                metadata={
                    "variant": variant_name,
                    "run_name": run_name,
                    "google_project_id": cfg.google_project_id,
                    "google_datastore_id": cfg.google_datastore_id,
                    "model_gemini": cfg.model_gemini,
                },
            )
            lf.score(trace_id=summary_trace.id, name="hit_rate_at_k", value=hit_rate)
            lf.score(trace_id=summary_trace.id, name="mrr", value=mrr)
            lf.score(
                trace_id=summary_trace.id, name="avg_latency_ms", value=avg_latency
            )
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
        config_snapshot={
            "retrieval_strategy": "google_adk",
            "google_project_id": cfg.google_project_id,
            "google_datastore_id": cfg.google_datastore_id,
            "model_gemini": cfg.model_gemini,
            "retrieval_k": cfg.retrieval_k,
            "embedding_provider": "google",
            "reranker_strategy": "n/a",
            "bm25_weight": "n/a",
            "vector_weight": "n/a",
        },
    )

    log.info(
        "experiment.google_adk.done",
        variant=variant_name,
        hit_rate=hit_rate,
        mrr=mrr,
        n=n,
        avg_latency_ms=round(avg_latency, 1),
    )
    return result


async def _run_adk_bedrock_experiment(
    variant_name: str,
    golden_samples: list[GoldenSample],
    *,
    cfg: LibrarySettings,
    run_name: str,
    ds_name: str,
    lf: Any | None,
) -> ExperimentResult:
    """Run experiment via ADK-wrapped Bedrock KB agent."""
    from orchestration.google_adk.bedrock_agent import BedrockKBAgent

    agent = BedrockKBAgent(cfg)
    query_results: list[QueryResult] = []

    for sample in golden_samples:
        try:
            ctx, _ = _build_adk_context(sample.query, sample.query_id)

            t0 = time.perf_counter()
            events = [e async for e in agent._run_async_impl(ctx)]
            latency_ms = (time.perf_counter() - t0) * 1000

            response_text = ""
            if events and events[0].content and events[0].content.parts:
                response_text = events[0].content.parts[0].text or ""

            urls = _extract_urls_from_adk_events(events)
            hit, rr = _score_hit(sample.expected_doc_url, urls)

            qr = QueryResult(
                query_id=sample.query_id,
                query=sample.query,
                hit=hit,
                reciprocal_rank=rr,
                retrieved_urls=urls[:5],
                expected_url=sample.expected_doc_url,
                latency_ms=latency_ms,
                answer=response_text,
            )
        except Exception as exc:
            log.warning(
                "experiment.adk_bedrock.query_failed",
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

    return _aggregate_results(
        query_results,
        variant_name=variant_name,
        ds_name=ds_name,
        run_name=run_name,
        lf=lf,
        config_snapshot={
            "retrieval_strategy": "adk_bedrock",
            "kb_id": cfg.bedrock_knowledge_base_id,
            "model_arn": cfg.bedrock_model_arn,
            "wrapper": "google-adk",
        },
    )


async def _run_adk_custom_rag_experiment(
    variant_name: str,
    golden_samples: list[GoldenSample],
    *,
    cfg: LibrarySettings,
    run_name: str,
    ds_name: str,
    lf: Any | None,
) -> ExperimentResult:
    """Run experiment via ADK agent with custom RAG tools (Gemini 2.0 Flash)."""
    from orchestration.google_adk.custom_rag_agent import run_custom_rag_query
    from orchestration.google_adk.factory import create_custom_rag

    agent = create_custom_rag(cfg)

    query_results: list[QueryResult] = []

    for sample in golden_samples:
        try:
            t0 = time.perf_counter()
            result = await run_custom_rag_query(
                agent, sample.query, session_id=f"eval-{sample.query_id}"
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            response_text = result.get("response", "")
            # Extract URLs from tool call events (search_knowledge_base returns urls)
            urls = _extract_urls_from_adk_events(result.get("events", []))
            hit, rr = _score_hit(sample.expected_doc_url, urls)

            qr = QueryResult(
                query_id=sample.query_id,
                query=sample.query,
                hit=hit,
                reciprocal_rank=rr,
                retrieved_urls=urls[:5],
                expected_url=sample.expected_doc_url,
                latency_ms=latency_ms,
                answer=response_text,
            )
        except Exception as exc:
            log.warning(
                "experiment.adk_custom_rag.query_failed",
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

    return _aggregate_results(
        query_results,
        variant_name=variant_name,
        ds_name=ds_name,
        run_name=run_name,
        lf=lf,
        config_snapshot={
            "retrieval_strategy": "adk_custom_rag",
            "model": "gemini-2.0-flash",
            "embedding_provider": cfg.embedding_provider,
            "reranker_strategy": cfg.reranker_strategy,
        },
    )


async def _run_adk_hybrid_experiment(
    variant_name: str,
    golden_samples: list[GoldenSample],
    *,
    cfg: LibrarySettings,
    run_name: str,
    ds_name: str,
    lf: Any | None,
) -> ExperimentResult:
    """Run experiment via ADK-wrapped full LangGraph pipeline."""
    from orchestration.google_adk.hybrid_agent import LibrarianADKAgent
    from orchestration.factory import create_librarian

    graph = create_librarian(cfg)
    agent = LibrarianADKAgent(graph=graph, cfg=cfg)
    query_results: list[QueryResult] = []

    for sample in golden_samples:
        try:
            ctx, _ = _build_adk_context(sample.query, sample.query_id)

            t0 = time.perf_counter()
            events = [e async for e in agent._run_async_impl(ctx)]
            latency_ms = (time.perf_counter() - t0) * 1000

            response_text = ""
            if events and events[0].content and events[0].content.parts:
                response_text = events[0].content.parts[0].text or ""

            urls = _extract_urls_from_adk_events(events)
            hit, rr = _score_hit(sample.expected_doc_url, urls)

            qr = QueryResult(
                query_id=sample.query_id,
                query=sample.query,
                hit=hit,
                reciprocal_rank=rr,
                retrieved_urls=urls[:5],
                expected_url=sample.expected_doc_url,
                latency_ms=latency_ms,
                answer=response_text,
            )
        except Exception as exc:
            log.warning(
                "experiment.adk_hybrid.query_failed",
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

    return _aggregate_results(
        query_results,
        variant_name=variant_name,
        ds_name=ds_name,
        run_name=run_name,
        lf=lf,
        config_snapshot={
            "retrieval_strategy": "adk_hybrid",
            "embedding_provider": cfg.embedding_provider,
            "reranker_strategy": cfg.reranker_strategy,
            "wrapper": "google-adk + langgraph",
        },
    )


async def run_variant_experiment(
    variant_name: str,
    golden_samples: list[GoldenSample],
    corpus: list[Chunk],
    *,
    cfg: LibrarySettings | None = None,
    dataset_name: str | None = None,
    langfuse_client: Any | None = None,
) -> ExperimentResult:
    """Run a single retrieval variant against golden samples and score results.

    Uses InMemoryRetriever populated with *corpus* for mock variants, or
    a real managed-RAG API when ``retrieval_strategy`` is ``"bedrock"`` or
    ``"google_adk"``.  Logs traces + scores to LangFuse when available.

    Args:
        variant_name: Key from VARIANTS (librarian, raptor, bedrock, bedrock-live, google-adk).
        golden_samples: Golden queries with expected doc URLs.
        corpus: Chunks to populate InMemoryRetriever (ignored for live variants).
        cfg: Variant config override (defaults to VARIANTS[variant_name]).
        dataset_name: LangFuse dataset name for linking.
        langfuse_client: Optional LangFuse client override.

    Returns:
        ExperimentResult with per-query and aggregate metrics.
    """
    cfg = cfg or VARIANTS[variant_name]
    ds_name = dataset_name or settings.langfuse_dataset_name
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_name = f"{variant_name}_{timestamp}"
    lf = langfuse_client or _get_langfuse_client()

    # Dispatch to real managed RAG APIs when configured
    if cfg.retrieval_strategy == "bedrock":
        return await _run_bedrock_experiment(
            variant_name,
            golden_samples,
            cfg=cfg,
            run_name=run_name,
            ds_name=ds_name,
            lf=lf,
        )
    if cfg.retrieval_strategy == "google_adk":
        return await _run_google_adk_experiment(
            variant_name,
            golden_samples,
            cfg=cfg,
            run_name=run_name,
            ds_name=ds_name,
            lf=lf,
        )
    if cfg.retrieval_strategy == "adk_bedrock":
        return await _run_adk_bedrock_experiment(
            variant_name,
            golden_samples,
            cfg=cfg,
            run_name=run_name,
            ds_name=ds_name,
            lf=lf,
        )
    if cfg.retrieval_strategy == "adk_custom_rag":
        return await _run_adk_custom_rag_experiment(
            variant_name,
            golden_samples,
            cfg=cfg,
            run_name=run_name,
            ds_name=ds_name,
            lf=lf,
        )
    if cfg.retrieval_strategy == "adk_hybrid":
        return await _run_adk_hybrid_experiment(
            variant_name,
            golden_samples,
            cfg=cfg,
            run_name=run_name,
            ds_name=ds_name,
            lf=lf,
        )

    from tests.librarian.testing.mock_embedder import MockEmbedder
    from storage.vectordb.inmemory import InMemoryRetriever

    # Build retriever with variant config
    embedder = MockEmbedder(dim=64, seed=42)
    retriever = InMemoryRetriever(
        bm25_weight=cfg.bm25_weight,
        vector_weight=cfg.vector_weight,
    )
    chunks_with_embeddings = [
        chunk.model_copy(update={"embedding": embedder.embed_passage(chunk.text)})
        for chunk in corpus
    ]
    await retriever.upsert(chunks_with_embeddings)

    k = cfg.retrieval_k
    clusterer = FailureClusterer()
    query_results: list[QueryResult] = []

    for sample in golden_samples:
        t0 = time.perf_counter()
        vec = embedder.embed_query(sample.query)
        results = await retriever.search(query_text=sample.query, query_vector=vec, k=k)
        latency_ms = (time.perf_counter() - t0) * 1000

        urls = [
            r.chunk.metadata.url if hasattr(r.chunk, "metadata") else ""
            for r in results
        ]
        hit = sample.expected_doc_url in urls
        rr = next(
            (1.0 / (i + 1) for i, u in enumerate(urls) if u == sample.expected_doc_url),
            0.0,
        )

        qr = QueryResult(
            query_id=sample.query_id,
            query=sample.query,
            hit=hit,
            reciprocal_rank=rr,
            retrieved_urls=urls[:5],
            expected_url=sample.expected_doc_url,
            latency_ms=latency_ms,
        )

        # Log to LangFuse
        if lf is not None:
            qr.trace_id = _langfuse_log_trace(lf, variant_name, qr, run_name, ds_name)

        query_results.append(qr)

    # Aggregate
    n = len(query_results)
    hits = sum(1 for qr in query_results if qr.hit)
    hit_rate = hits / n if n else 0.0
    mrr = sum(qr.reciprocal_rank for qr in query_results) / n if n else 0.0
    avg_latency = sum(qr.latency_ms for qr in query_results) / n if n else 0.0

    # Cluster failures using the existing FailureClusterer
    from librarian.ingestion.tasks.tracing import PipelineTrace, PipelineTracer

    tracer = PipelineTracer()
    for qr in query_results:
        trace = tracer.create_trace(qr.query_id, qr.query)
        trace.status = "success" if qr.hit else "failure"
        trace.confidence = qr.reciprocal_rank
        if not qr.hit:
            trace.failure_reason = "expected_doc_not_in_top_k"
    clusters = clusterer.cluster_failures(tracer.get_failure_traces())

    if lf is not None:
        # Log aggregate scores as a summary trace
        try:
            summary_trace = lf.trace(
                name=f"experiment_summary_{variant_name}",
                metadata={
                    "variant": variant_name,
                    "run_name": run_name,
                    "k": k,
                    "embedding_model": cfg.embedding_model,
                    "reranker_strategy": cfg.reranker_strategy,
                    "bm25_weight": cfg.bm25_weight,
                    "vector_weight": cfg.vector_weight,
                },
            )
            lf.score(trace_id=summary_trace.id, name="hit_rate_at_k", value=hit_rate)
            lf.score(trace_id=summary_trace.id, name="mrr", value=mrr)
            lf.score(
                trace_id=summary_trace.id, name="avg_latency_ms", value=avg_latency
            )
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
        config_snapshot={
            "embedding_model": cfg.embedding_model,
            "embedding_provider": cfg.embedding_provider,
            "reranker_strategy": cfg.reranker_strategy,
            "retrieval_k": cfg.retrieval_k,
            "bm25_weight": cfg.bm25_weight,
            "vector_weight": cfg.vector_weight,
        },
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


async def run_all_experiments(
    golden_samples: list[GoldenSample],
    corpus: list[Chunk],
    *,
    variants: dict[str, LibrarySettings] | None = None,
    dataset_name: str | None = None,
    langfuse_client: Any | None = None,
) -> dict[str, ExperimentResult]:
    """Run all variants and return comparison results.

    Args:
        golden_samples: Golden queries.
        corpus: Chunks for InMemoryRetriever.
        variants: Override variant configs (defaults to VARIANTS).
        dataset_name: LangFuse dataset name.
        langfuse_client: Optional LangFuse client.

    Returns:
        Dict mapping variant name to ExperimentResult.
    """
    variant_configs = variants or VARIANTS
    results: dict[str, ExperimentResult] = {}

    for name, cfg in variant_configs.items():
        # Skip live variants when credentials are absent
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
        if (
            cfg.retrieval_strategy == "adk_bedrock"
            and not cfg.bedrock_knowledge_base_id
        ):
            log.info(
                "experiment.variant.skipped",
                variant=name,
                reason="BEDROCK_KNOWLEDGE_BASE_ID not set (adk_bedrock)",
            )
            continue
        if cfg.retrieval_strategy == "adk_custom_rag" and not cfg.gemini_api_key:
            log.info(
                "experiment.variant.skipped",
                variant=name,
                reason="GEMINI_API_KEY not set (adk_custom_rag)",
            )
            continue

        results[name] = await run_variant_experiment(
            name,
            golden_samples,
            corpus,
            cfg=cfg,
            dataset_name=dataset_name,
            langfuse_client=langfuse_client,
        )

    return results


# ---------------------------------------------------------------------------
# Rich output
# ---------------------------------------------------------------------------


def print_comparison_table(results: dict[str, ExperimentResult]) -> None:
    """Print a formatted comparison table to stdout."""
    header = (
        f"\n{'Variant':<12} {'hit_rate':>10} {'MRR':>10} {'n':>5} "
        f"{'hits':>5} {'avg_ms':>8} {'failures'}"
    )
    sep = "-" * 80
    print(f"\n{sep}")  # noqa: T201
    print("  Experiment Comparison")  # noqa: T201
    print(sep)  # noqa: T201
    print(header)  # noqa: T201
    print(sep)  # noqa: T201

    for name, r in results.items():
        failure_str = (
            ", ".join(f"{c.failure_type}×{c.count}" for c in r.failure_clusters)
            or "none"
        )
        print(  # noqa: T201
            f"  {name:<10} {r.hit_rate:>10.3f} {r.mrr:>10.3f} {r.n_queries:>5} "
            f"{r.n_hits:>5} {r.avg_latency_ms:>7.1f} [{failure_str}]"
        )

    print(sep)  # noqa: T201

    # Config snapshot for each variant
    print("\n  Configuration:")  # noqa: T201
    for name, r in results.items():
        cs = r.config_snapshot
        if cs.get("retrieval_strategy") == "bedrock":
            kb_id = cs.get("kb_id", "?")
            kb_display = f"{kb_id[:12]}..." if len(kb_id) > 12 else kb_id
            print(  # noqa: T201
                f"    {name}: bedrock-kb kb={kb_display} k={cs.get('retrieval_k', '?')}"
            )
        elif cs.get("retrieval_strategy") == "google_adk":
            ds_id = cs.get("google_datastore_id", "?")
            ds_display = f"{ds_id[:12]}..." if len(ds_id) > 12 else ds_id
            print(  # noqa: T201
                f"    {name}: gemini+vertex "
                f"ds={ds_display} "
                f"model={cs.get('model_gemini', '?')} "
                f"k={cs.get('retrieval_k', '?')}"
            )
        else:
            print(  # noqa: T201
                f"    {name}: {cs.get('embedding_provider', '?')} "
                f"k={cs.get('retrieval_k', '?')} "
                f"reranker={cs.get('reranker_strategy', '?')} "
                f"bm25={cs.get('bm25_weight', '?')}/{cs.get('vector_weight', '?')}"
            )

    # LangFuse status
    if settings.langfuse_enabled:
        print(f"\n  LangFuse: traces logged → {settings.langfuse_host}")  # noqa: T201
    else:
        print(  # noqa: T201
            "\n  LangFuse: disabled — set LANGFUSE_ENABLED=true to log traces"
        )
    print()  # noqa: T201


def export_results(
    results: dict[str, ExperimentResult],
    output_path: str | Path,
) -> Path:
    """Export experiment results to JSON for the eval dashboard.

    Args:
        results: Dict mapping variant name to ExperimentResult.
        output_path: File path for the JSON output.

    Returns:
        Resolved path of the written file.
    """
    path = Path(output_path)
    timestamp = datetime.now().isoformat()

    payload: dict[str, Any] = {
        "exported_at": timestamp,
        "variants": {},
    }

    for name, result in results.items():
        query_data = [
            {
                "query_id": qr.query_id,
                "query": qr.query,
                "hit": qr.hit,
                "reciprocal_rank": qr.reciprocal_rank,
                "retrieved_urls": qr.retrieved_urls,
                "expected_url": qr.expected_url,
                "latency_ms": qr.latency_ms,
                "trace_id": qr.trace_id,
            }
            for qr in result.query_results
        ]

        failure_data = [
            {
                "failure_type": c.failure_type,
                "count": c.count,
                "common_patterns": c.common_patterns,
            }
            for c in result.failure_clusters
        ]

        payload["variants"][name] = {
            **result.summary_dict(),
            "dataset_name": result.dataset_name,
            "query_results": query_data,
            "failure_clusters": failure_data,
            "config_snapshot": result.config_snapshot,
        }

    path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("experiment.export.done", path=str(path), n_variants=len(results))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_samples(path: str | None = None) -> list[GoldenSample]:
    """Load golden samples from the configured or provided path."""
    dataset_path = path or settings.eval_dataset_path
    if not dataset_path:
        # Fall back to the test corpus
        log.info(
            "experiment.load.fallback",
            msg="No EVAL_DATASET_PATH set — using built-in test samples",
        )
        from tests.librarian.evalsuite.conftest import GOLDEN

        return GOLDEN

    log.info("experiment.load", path=dataset_path)
    return load_golden_from_jsonl(dataset_path)


def _load_corpus(samples: list[GoldenSample]) -> list[Chunk]:
    """Load a corpus aligned with the golden samples.

    For external datasets, builds minimal placeholder chunks from the
    golden sample metadata so InMemoryRetriever has something to index.
    For built-in test samples, uses the test corpus.
    """
    # Check if these are the built-in test samples
    if samples and samples[0].query_id == "q1":
        from tests.librarian.evalsuite.conftest import CORPUS

        return CORPUS

    # For external golden samples, create placeholder chunks from metadata
    # so that the expected_doc_url can be matched.
    # In production, the retriever would search a real vector store.
    chunks: list[Chunk] = []
    seen_urls: set[str] = set()
    for sample in samples:
        url = sample.expected_doc_url
        if url and url not in seen_urls:
            seen_urls.add(url)
            chunks.append(
                Chunk(
                    id=f"placeholder_{len(chunks)}",
                    text=sample.query,
                    metadata=ChunkMetadata(
                        url=url, title=url, doc_id=f"placeholder_{len(chunks)}"
                    ),
                )
            )
    return chunks


def _cli_upload(args: list[str]) -> None:
    """CLI: upload golden dataset to LangFuse."""
    import argparse

    parser = argparse.ArgumentParser(description="Upload golden dataset to LangFuse")
    parser.add_argument(
        "--path", help="Path to golden JSONL (overrides EVAL_DATASET_PATH)"
    )
    parser.add_argument("--dataset", help="LangFuse dataset name", default=None)
    parsed = parser.parse_args(args)

    samples = _load_samples(parsed.path)
    n = upload_golden_dataset(samples, dataset_name=parsed.dataset)
    if n:
        print(
            f"Uploaded {n} samples to LangFuse dataset '{parsed.dataset or settings.langfuse_dataset_name}'"
        )  # noqa: T201
    else:
        print("Upload skipped — check LANGFUSE_ENABLED and credentials")  # noqa: T201


def _cli_run(args: list[str]) -> None:
    """CLI: run experiment(s)."""
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
        "--export",
        help="Export results to JSON file for the eval dashboard",
        default=None,
        metavar="FILE",
    )
    parsed = parser.parse_args(args)

    samples = _load_samples(parsed.path)
    corpus = _load_corpus(samples)
    dataset_name = parsed.dataset or settings.langfuse_dataset_name

    if parsed.variant:
        if parsed.variant not in VARIANTS:
            print(
                f"Unknown variant '{parsed.variant}'. Available: {list(VARIANTS.keys())}"
            )  # noqa: T201
            sys.exit(1)
        result = asyncio.run(
            run_variant_experiment(
                parsed.variant,
                samples,
                corpus,
                dataset_name=dataset_name,
            )
        )
        all_results = {parsed.variant: result}
        print_comparison_table(all_results)
    else:
        all_results = asyncio.run(
            run_all_experiments(
                samples,
                corpus,
                dataset_name=dataset_name,
            )
        )
        print_comparison_table(all_results)

    if parsed.export:
        out = export_results(all_results, parsed.export)
        print(f"Results exported to {out}")  # noqa: T201


def main() -> None:
    """CLI entrypoint: ``python -m eval.experiment <upload|run> [args]``."""
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python -m eval.experiment <upload|run> [options]")  # noqa: T201
        print()  # noqa: T201
        print("Commands:")  # noqa: T201
        print("  upload  Upload golden dataset to LangFuse")  # noqa: T201
        print("  run     Run variant experiments (all or single)")  # noqa: T201
        print()  # noqa: T201
        print("Examples:")  # noqa: T201
        print("  python -m eval.experiment upload --path /data/golden.jsonl")  # noqa: T201
        print("  python -m eval.experiment run")  # noqa: T201
        print("  python -m eval.experiment run --variant librarian")  # noqa: T201
        print("  python -m eval.experiment run --export results.json")  # noqa: T201
        sys.exit(0)

    command = sys.argv[1]
    remaining = sys.argv[2:]

    if command == "upload":
        _cli_upload(remaining)
    elif command == "run":
        _cli_run(remaining)
    else:
        print(f"Unknown command '{command}'. Use 'upload' or 'run'.")  # noqa: T201
        sys.exit(1)


if __name__ == "__main__":
    main()
