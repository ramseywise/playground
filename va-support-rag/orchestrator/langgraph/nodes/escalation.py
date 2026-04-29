"""Escalation node (LangGraph id: ``escalation``)."""

from __future__ import annotations

import logging

from orchestrator.langgraph.schemas.state import GraphState
from orchestrator.langgraph.utils import with_total_ms

log = logging.getLogger(__name__)


def escalation_node(state: GraphState) -> dict:
    reason = state.escalation_reason or "unspecified"
    lines = [
        "We could not generate a reliable answer automatically.",
        f"Reason: {reason}.",
        "",
        "Please try rephrasing your question or contact support for help.",
    ]
    if state.market:
        lines.insert(2, f"Market: {state.market}.")
    text = "\n".join(lines)
    lat = with_total_ms(state.latency_ms)
    log.info("escalation: reason=%s total_ms=%.1f", reason, lat.get("total_ms", 0.0))
    return {
        "final_answer": text,
        "citations": [],
        "qa_outcome": "escalate",
        "latency_ms": lat,
    }


__all__ = ["escalation_node"]
