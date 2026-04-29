"""Post-rerank confidence routing node (LangGraph id: ``qa_policy_rerank``)."""

from __future__ import annotations

import logging
import time

from core.config import RAG_CONFIDENCE_THRESHOLD, RAG_POLICY_HYBRID_BORDER_LOW
from orchestrator.langgraph.policies.confidence_routing import decide_after_rerank
from orchestrator.langgraph.policies.hybrid_policy import maybe_upgrade_rerank_route
from orchestrator.langgraph.schemas.state import GraphState

log = logging.getLogger(__name__)


def qa_policy_rerank_node(state: GraphState) -> dict:
    t0 = time.perf_counter()
    route, reason = decide_after_rerank(
        error=state.error,
        reranked_chunks=list(state.reranked_chunks),
        confidence_score=state.confidence_score,
        threshold=RAG_CONFIDENCE_THRESHOLD,
    )
    route, reason = maybe_upgrade_rerank_route(
        route=route,
        reason=reason,
        reranked_chunks=list(state.reranked_chunks),
        confidence_score=state.confidence_score,
        threshold=RAG_CONFIDENCE_THRESHOLD,
        border_low_frac=RAG_POLICY_HYBRID_BORDER_LOW,
        query=state.query or "",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "qa_policy_rerank decision=%s reason=%s rerank_top_score=%.4f rerank_threshold=%.4f elapsed_ms=%.1f",
        route,
        reason,
        state.confidence_score,
        RAG_CONFIDENCE_THRESHOLD,
        elapsed_ms,
    )
    out: dict = {
        "qa_after_rerank": route,
        "latency_ms": {**state.latency_ms, "policy_rerank_ms": elapsed_ms},
    }
    if route == "answer":
        out["qa_outcome"] = "answer"
        out["escalation_reason"] = None
    elif route == "gate":
        out["qa_outcome"] = None
        out["escalation_reason"] = reason
    else:
        out["qa_outcome"] = "escalate"
        out["escalation_reason"] = reason
    return out


__all__ = ["qa_policy_rerank_node"]
