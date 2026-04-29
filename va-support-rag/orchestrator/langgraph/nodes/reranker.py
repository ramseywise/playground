"""Reranking node (LangGraph id: ``reranker``)."""

from __future__ import annotations

import logging
import time

from core.config import RERANKER_TOP_K
from orchestrator.langgraph.schemas.state import GraphState
from orchestrator.langgraph.utils import run_coro
from rag.retrieval.pipeline import NO_CHUNKS_CONFIDENCE, rerank_graded_chunks

log = logging.getLogger(__name__)


def reranker_node(state: GraphState) -> dict:
    t0 = time.perf_counter()
    graded = list(state.graded_chunks)
    query = (state.retrieval_queries[0] if state.retrieval_queries else None) or (
        state.query or ""
    )

    if not graded:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info("reranker: skip stage=rerank reason=no_graded_chunks")
        return {
            "reranked_chunks": [],
            "confidence_score": NO_CHUNKS_CONFIDENCE,
            "latency_ms": {**state.latency_ms, "rerank_ms": elapsed_ms},
        }

    try:
        ranked = run_coro(rerank_graded_chunks(query, graded, RERANKER_TOP_K))
    except Exception:
        log.exception("reranker.failed")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "reranked_chunks": [],
            "confidence_score": NO_CHUNKS_CONFIDENCE,
            "error": "Rerank failed (see logs)",
            "latency_ms": {**state.latency_ms, "rerank_ms": elapsed_ms},
        }

    confidence = max((r.relevance_score for r in ranked), default=NO_CHUNKS_CONFIDENCE)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "reranker: done stage=rerank candidates=%d reranked=%d rerank_top_score=%.4f elapsed_ms=%.1f",
        len(graded),
        len(ranked),
        confidence,
        elapsed_ms,
    )
    return {
        "reranked_chunks": ranked,
        "confidence_score": confidence,
        "error": None,
        "latency_ms": {**state.latency_ms, "rerank_ms": elapsed_ms},
    }


__all__ = ["reranker_node"]
