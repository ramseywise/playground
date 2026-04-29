"""Retrieval HITL gate node (LangGraph id: ``qa_retrieval_gate``)."""

from __future__ import annotations

import logging

from langgraph.types import interrupt

from orchestrator.langgraph.policies.confidence_routing import retrieval_signal
from orchestrator.langgraph.routing import parse_retrieval_gate_decision
from orchestrator.langgraph.schemas.state import GraphState

log = logging.getLogger(__name__)


def qa_retrieval_gate_node(state: GraphState) -> dict:
    reason = state.escalation_reason or "retrieval_quality"
    conf = retrieval_signal(list(state.graded_chunks))
    log.info(
        "qa_retrieval_gate: interrupt reason=%s ensemble_signal=%.4f", reason, conf
    )
    raw = interrupt(
        {
            "kind": "qa_retrieval_gate",
            "reason": reason,
            "retrieval_confidence": conf,
            "hint": (
                "Choose: action=refine with optional query, action=continue to rerank, "
                "or action=escalate for human support."
            ),
        }
    )
    target, refined_query = parse_retrieval_gate_decision(raw)
    out: dict = {"qa_retrieval_gate_action": target}
    if target == "reranker":
        out["qa_after_retrieval"] = None
    if target == "escalation":
        out["escalation_reason"] = "user_escalation_after_retrieval"
    if refined_query:
        out["query"] = refined_query
    log.info("qa_retrieval_gate: resolved target=%s", target)
    return out


__all__ = ["qa_retrieval_gate_node"]
