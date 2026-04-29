"""Optional post-answer evaluation (``RAG_POST_ANSWER_EVALUATOR``)."""

from __future__ import annotations

import logging
import time

from core.config import RAG_POST_ANSWER_EVALUATOR
from orchestrator.langgraph.schemas.state import qa_cycle_field_reset
from orchestrator.langgraph.schemas.state import GraphState
from clients import llm as llm_mod
from orchestrator.langgraph.chains import get_post_answer_eval_chain

log = logging.getLogger(__name__)


def post_answer_evaluator_node(state: GraphState) -> dict:
    if not RAG_POST_ANSWER_EVALUATOR:
        return {}
    if state.mode != "q&a":
        return {}
    draft = state.final_answer
    if not isinstance(draft, str) or not draft.strip():
        return {}
    if not llm_mod.llm_configured():
        log.warning("post_answer_eval.skip reason=llm_not_configured")
        return {}

    t0 = time.perf_counter()
    try:
        ids = ",".join(c.chunk_id for c in state.citations) or "(none)"
        out = get_post_answer_eval_chain().invoke(
            {"query": state.query or "", "draft": draft, "citation_ids": ids}
        )
    except Exception:
        log.exception("post_answer_eval.failed")
        return {}

    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info("post_answer_eval verdict=%s elapsed_ms=%.1f", out.verdict, elapsed_ms)

    lat = {**state.latency_ms, "post_answer_eval_ms": elapsed_ms}

    if out.verdict == "accept":
        return {"latency_ms": lat}

    if out.verdict == "escalate":
        msg = (out.public_message or "").strip() or (
            "We could not verify this answer automatically. Please contact support."
        )
        return {
            "final_answer": msg,
            "qa_outcome": "escalate",
            "escalation_reason": "post_answer_evaluator",
            "citations": [],
            "latency_ms": lat,
        }

    if out.verdict == "refine":
        q = (out.refinement_query or state.query or "").strip()
        upd = qa_cycle_field_reset()
        upd.update(
            {
                "query": q,
                "final_answer": None,
                "citations": [],
                "qa_post_answer_branch": "retriever",
                "latency_ms": lat,
            }
        )
        return upd

    return {}


__all__ = ["post_answer_evaluator_node"]
