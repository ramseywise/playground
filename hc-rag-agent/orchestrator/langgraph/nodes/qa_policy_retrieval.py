"""Post-retrieval confidence routing node (LangGraph id: ``qa_policy_retrieval``)."""

from __future__ import annotations

import logging
import time

from core.config import (
    RAG_ENSEMBLE_SCORE_THRESHOLD,
    RAG_POLICY_HYBRID_BORDER_LOW,
)
from orchestrator.langgraph.policies.confidence_routing import (
    decide_after_retrieval,
    retrieval_signal,
)
from orchestrator.langgraph.policies.hybrid_policy import (
    maybe_upgrade_retrieval_route,
)
from orchestrator.langgraph.schemas.state import GraphState

log = logging.getLogger(__name__)


def qa_policy_retrieval_node(state: GraphState) -> dict:
    t0 = time.perf_counter()
    route, reason = decide_after_retrieval(
        error=state.error,
        graded_chunks=list(state.graded_chunks),
        ensemble_threshold=RAG_ENSEMBLE_SCORE_THRESHOLD,
    )
    route, reason = maybe_upgrade_retrieval_route(
        route=route,
        reason=reason,
        graded_chunks=list(state.graded_chunks),
        ensemble_threshold=RAG_ENSEMBLE_SCORE_THRESHOLD,
        border_low_frac=RAG_POLICY_HYBRID_BORDER_LOW,
        query=state.query or "",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    sig = retrieval_signal(list(state.graded_chunks))
    log.info(
        "qa_policy_retrieval decision=%s reason=%s ensemble_signal=%.4f ensemble_threshold=%.4f elapsed_ms=%.1f",
        route,
        reason,
        sig,
        RAG_ENSEMBLE_SCORE_THRESHOLD,
        elapsed_ms,
    )
    return {
        "qa_after_retrieval": route,
        "escalation_reason": reason,
        "latency_ms": {**state.latency_ms, "policy_retrieval_ms": elapsed_ms},
    }


__all__ = ["qa_policy_retrieval_node"]
