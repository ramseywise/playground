"""Routing logic — conditional edges for the Q&A retrieval pipeline."""

from __future__ import annotations

import logging
from typing import Literal, Tuple

from orchestrator.langgraph.schemas.state import GraphState

log = logging.getLogger(__name__)

RetrievalGateTarget = Literal["retriever", "reranker", "escalation"]
RerankGateTarget = Literal["retriever", "answer", "escalation"]


def parse_retrieval_gate_decision(
    raw: object,
) -> Tuple[RetrievalGateTarget, str | None]:
    """Map interrupt resume value to next graph node; optional refined query for retriever."""
    if isinstance(raw, dict):
        action = str(raw.get("action", "")).strip().lower()
        q = raw.get("query")
        q_str = q.strip() if isinstance(q, str) else None
        if action in ("refine", "refine_query", "retry", "rephrase"):
            return "retriever", q_str
        if action in ("continue", "rerank", "proceed", "try_rerank", "yes", "ok"):
            return "reranker", None
        if action in ("escalate", "human", "handoff", "support", "agent"):
            return "escalation", None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"refine", "r", "retry", "rephrase"}:
            return "retriever", None
        if s in {"continue", "c", "rerank", "yes", "y", "ok"}:
            return "reranker", None
        if s in {"escalate", "e", "human", "handoff"}:
            return "escalation", None
    return "escalation", None


def parse_rerank_gate_decision(raw: object) -> Tuple[RerankGateTarget, str | None]:
    """Map interrupt resume value after marginal rerank scores."""
    if isinstance(raw, dict):
        action = str(raw.get("action", "")).strip().lower()
        q = raw.get("query")
        q_str = q.strip() if isinstance(q, str) else None
        if action in ("refine", "refine_query", "retry", "rephrase"):
            return "retriever", q_str
        if action in ("answer", "answer_anyway", "use_anyway", "generate", "yes", "ok"):
            return "answer", None
        if action in ("escalate", "human", "handoff", "support", "agent"):
            return "escalation", None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"refine", "r", "retry"}:
            return "retriever", None
        if s in {"answer", "a", "yes", "y", "ok", "generate"}:
            return "answer", None
        if s in {"escalate", "e", "human", "handoff"}:
            return "escalation", None
    return "escalation", None


def route_after_qa_policy_retrieval(state: GraphState) -> str:
    """After ensemble signal + confidence routing — rerank, HITL gate, or auto-escalate."""
    r = state.qa_after_retrieval
    if r == "rerank":
        return "rerank"
    if r == "gate":
        return "gate"
    return "escalate"


def route_after_qa_retrieval_gate(state: GraphState) -> str:
    """Resume from retrieval HITL — retriever (refine), reranker (continue), or escalation."""
    a = state.qa_retrieval_gate_action
    if a == "retriever":
        return "retriever"
    if a == "reranker":
        return "reranker"
    return "escalate"


def route_after_qa_policy_rerank(state: GraphState) -> str:
    """After rerank top score + confidence routing — answer, HITL gate, or auto-escalate."""
    r = state.qa_after_rerank
    if r == "answer":
        return "answer"
    if r == "gate":
        return "gate"
    return "escalate"


def route_after_qa_rerank_gate(state: GraphState) -> str:
    """Resume from rerank HITL — retriever (refine), answer (force), or escalation."""
    a = state.qa_rerank_gate_action
    if a == "retriever":
        return "retriever"
    if a == "answer":
        return "answer"
    return "escalate"


def route_to_summarizer(state: GraphState) -> str:
    """Always routes to END — the summarizer node is a terminal step."""
    return "end"


def route_after_post_answer(state: GraphState) -> str:
    """After optional post-answer evaluation — refine loops to retriever or end."""
    from core.config import RAG_POST_ANSWER_EVALUATOR

    if not RAG_POST_ANSWER_EVALUATOR:
        return "end"
    if state.qa_post_answer_branch == "retriever":
        return "retriever"
    return "end"
