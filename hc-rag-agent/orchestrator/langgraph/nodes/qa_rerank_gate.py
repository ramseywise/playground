"""Rerank HITL gate node (LangGraph id: ``qa_rerank_gate``)."""

from __future__ import annotations

import logging

from langgraph.types import interrupt

from orchestrator.langgraph.routing import parse_rerank_gate_decision
from orchestrator.langgraph.schemas.state import GraphState

log = logging.getLogger(__name__)


def qa_rerank_gate_node(state: GraphState) -> dict:
    reason = state.escalation_reason or "rerank_quality"
    log.info(
        "qa_rerank_gate: interrupt reason=%s rerank_top_score=%.4f",
        reason,
        state.confidence_score,
    )
    raw = interrupt(
        {
            "kind": "qa_rerank_gate",
            "reason": reason,
            "rerank_confidence": state.confidence_score,
            "hint": (
                "Choose: action=refine with optional query, action=answer to generate anyway, "
                "or action=escalate for human support."
            ),
        }
    )
    target, refined_query = parse_rerank_gate_decision(raw)
    out: dict = {"qa_rerank_gate_action": target}
    if target == "answer":
        out["qa_after_rerank"] = None
        out["qa_outcome"] = "answer"
    if target == "escalation":
        out["escalation_reason"] = "user_escalation_after_rerank"
    if refined_query:
        out["query"] = refined_query
    log.info("qa_rerank_gate: resolved target=%s", target)
    return out


__all__ = ["qa_rerank_gate_node"]
