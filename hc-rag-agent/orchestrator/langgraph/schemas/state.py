from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field

from orchestrator.langgraph.schemas.contract import Citation, QAOutcome
from rag.schemas.chunks import GradedChunk, RankedChunk

_QA_CYCLE_RESET: Dict[str, Any] = {
    "qa_after_retrieval": None,
    "qa_after_rerank": None,
    "qa_retrieval_gate_action": None,
    "qa_rerank_gate_action": None,
    "qa_post_answer_branch": None,
    "qa_outcome": None,
    "escalation_reason": None,
    "reranked_chunks": [],
    "confidence_score": 0.0,
}


def qa_cycle_field_reset() -> Dict[str, Any]:
    """Return a copy of state updates that clear a Q&A retrieval cycle."""
    return dict(_QA_CYCLE_RESET)


class GraphState(BaseModel):
    # conversational context
    messages: List[AnyMessage] = Field(default_factory=list)
    query: Optional[str] = None

    # planner
    mode: Optional[Literal["q&a", "task_execution"]] = None
    planner_intent: Optional[str] = Field(
        default=None,
        description="Short intent label from LLM planner (optional).",
    )
    planner_retrieval_hints: List[str] = Field(
        default_factory=list,
        description="Optional retrieval hints from LLM planner for q&a.",
    )

    # -------- q&a (contract fields) --------
    market: Optional[str] = None
    locale: Optional[str] = None
    qa_outcome: Optional[QAOutcome] = None
    qa_after_retrieval: Optional[Literal["rerank", "gate", "escalate"]] = Field(
        default=None,
        description="Post-retrieval policy branch: rerank, HITL gate, or auto-escalate.",
    )
    qa_after_rerank: Optional[Literal["answer", "gate", "escalate"]] = Field(
        default=None,
        description="Post-rerank policy branch: answer, HITL gate, or auto-escalate.",
    )
    qa_retrieval_gate_action: Optional[
        Literal["retriever", "reranker", "escalation"]
    ] = Field(
        default=None,
        description="User choice after qa_retrieval_gate interrupt (routing key).",
    )
    qa_rerank_gate_action: Optional[Literal["retriever", "answer", "escalation"]] = (
        Field(
            default=None,
            description="User choice after qa_rerank_gate interrupt (routing key).",
        )
    )
    citations: List[Citation] = Field(default_factory=list)
    qa_post_answer_branch: Optional[Literal["end", "retriever"]] = Field(
        default=None,
        description="Set by post_answer_evaluator when refine routes back to retriever.",
    )
    escalation_reason: Optional[str] = None
    latency_ms: Dict[str, float] = Field(
        default_factory=dict,
        description="Structured latency (ms) per pipeline stage and total.",
    )

    retrieved_context: Optional[str] = None
    retrieval_artifacts: Optional[List[Dict[str, Any]]] = None
    retrieval_queries: Optional[List[str]] = Field(
        default=None,
        description="Queries sent to the retriever (e.g. German variants); user-facing query stays in `query`.",
    )

    graded_chunks: List[GradedChunk] = Field(default_factory=list)
    reranked_chunks: List[RankedChunk] = Field(default_factory=list)
    confidence_score: float = 0.0

    # -------- task execution --------
    collected_fields: Dict[str, Any] = Field(default_factory=dict)
    clarification_history: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    clarify_round: int = 0
    max_clarify_rounds: int = 2
    scheduler_confirmed: Optional[bool] = None
    action_steps: List[str] = Field(default_factory=list)

    # --- session bucket (product may add summarization / prefs here) ---
    session_memory: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional session-scoped key/value store; not populated by default.",
    )

    error: Optional[str] = None
    final_answer: Any = None


__all__ = ["GraphState", "_QA_CYCLE_RESET", "qa_cycle_field_reset"]
