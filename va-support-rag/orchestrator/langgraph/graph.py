"""LangGraph graph — deterministic Q&A retrieval subgraph.

Planner defaults to keyword routing (``RAG_LLM_PLANNER=false``); set ``RAG_LLM_PLANNER=true``
for the LLM planner.

Q&A path::

    START → planner → retriever → qa_policy_retrieval → reranker | qa_retrieval_gate | escalation
    qa_retrieval_gate → retriever | reranker | escalation
    reranker → qa_policy_rerank → answer | qa_rerank_gate | escalation
    qa_rerank_gate → retriever | answer | escalation
    answer → post_answer_evaluator → summarizer → END

**Checkpointer**

``poc_graph`` uses ``MemorySaver`` for the CLI runner and unit tests.

For the standalone HTTP API, pass an async checkpointer from the FastAPI lifespan:

    async with AsyncSqliteSaver.from_conn_string(...) as cp:
        graph = build_rag_subgraph(cp)

When embedded as a subgraph inside va-langgraph or va-google-adk, call
``build_rag_subgraph()`` with no arguments — the default MemorySaver is
sufficient for single-invocation Q&A calls.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from orchestrator.langgraph.nodes import (
    answer_node,
    escalation_node,
    planner_node,
    qa_policy_rerank_node,
    qa_policy_retrieval_node,
    qa_rerank_gate_node,
    qa_retrieval_gate_node,
    reranker_node,
    retriever_node,
)
from orchestrator.langgraph.nodes.post_answer_node import post_answer_evaluator_node
from orchestrator.langgraph.nodes.summarizer import summarizer_node
from orchestrator.langgraph.routing import (
    route_after_post_answer,
    route_after_qa_policy_rerank,
    route_after_qa_policy_retrieval,
    route_after_qa_retrieval_gate,
    route_after_qa_rerank_gate,
    route_to_summarizer,
)
from orchestrator.langgraph.schemas.state import GraphState


def _make_retrieval_state_graph() -> StateGraph:
    """Build retrieval-only graph: planner → retriever → qa_policy → reranker.

    Stops before answer synthesis. Used by VAs that inject context + answer themselves.
    Skips: answer, post_answer_evaluator, summarizer, and rerank gates.
    """
    g = StateGraph(GraphState)

    g.add_node("planner", planner_node)
    g.add_node("retriever", retriever_node)
    g.add_node("qa_policy_retrieval", qa_policy_retrieval_node)
    g.add_node("reranker", reranker_node)
    g.add_node("qa_policy_rerank", qa_policy_rerank_node)
    g.add_node("escalation", escalation_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")

    g.add_edge("retriever", "qa_policy_retrieval")

    g.add_conditional_edges(
        "qa_policy_retrieval",
        route_after_qa_policy_retrieval,
        {"rerank": "reranker", "gate": "escalation", "escalate": "escalation"},
    )

    g.add_edge("reranker", "qa_policy_rerank")

    g.add_conditional_edges(
        "qa_policy_rerank",
        route_after_qa_policy_rerank,
        {"answer": END, "gate": END, "escalate": "escalation"},
    )

    g.add_edge("escalation", END)

    return g


def _make_state_graph() -> StateGraph:
    """Build and return the uncompiled StateGraph (no checkpointer attached)."""
    g = StateGraph(GraphState)

    g.add_node("planner", planner_node)
    g.add_node("retriever", retriever_node)
    g.add_node("qa_policy_retrieval", qa_policy_retrieval_node)
    g.add_node("qa_retrieval_gate", qa_retrieval_gate_node)
    g.add_node("reranker", reranker_node)
    g.add_node("qa_policy_rerank", qa_policy_rerank_node)
    g.add_node("qa_rerank_gate", qa_rerank_gate_node)
    g.add_node("escalation", escalation_node)
    g.add_node("answer", answer_node)
    g.add_node("post_answer_evaluator", post_answer_evaluator_node)
    g.add_node("summarizer", summarizer_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")  # always Q&A — task-execution path removed

    g.add_edge("retriever", "qa_policy_retrieval")

    g.add_conditional_edges(
        "qa_policy_retrieval",
        route_after_qa_policy_retrieval,
        {"rerank": "reranker", "gate": "qa_retrieval_gate", "escalate": "escalation"},
    )

    g.add_conditional_edges(
        "qa_retrieval_gate",
        route_after_qa_retrieval_gate,
        {"retriever": "retriever", "reranker": "reranker", "escalate": "escalation"},
    )

    g.add_edge("reranker", "qa_policy_rerank")

    g.add_conditional_edges(
        "qa_policy_rerank",
        route_after_qa_policy_rerank,
        {"answer": "answer", "gate": "qa_rerank_gate", "escalate": "escalation"},
    )

    g.add_conditional_edges(
        "qa_rerank_gate",
        route_after_qa_rerank_gate,
        {"retriever": "retriever", "answer": "answer", "escalate": "escalation"},
    )

    g.add_edge("escalation", END)

    g.add_edge("answer", "post_answer_evaluator")
    g.add_conditional_edges(
        "post_answer_evaluator",
        route_after_post_answer,
        {"end": "summarizer", "retriever": "retriever"},
    )
    g.add_conditional_edges("summarizer", route_to_summarizer, {"end": END})

    return g


def build_retrieval_subgraph(checkpointer: Any = None) -> Any:
    """Compile and return retrieval-only subgraph (no answer synthesis).

    Used by VA agents (va-langgraph, va-google-adk) via POST /api/v1/retrieval.
    Returns structured documents + confidence for the VA to synthesize an answer.
    """
    cp = checkpointer if checkpointer is not None else MemorySaver()
    return _make_retrieval_state_graph().compile(checkpointer=cp)


def build_rag_subgraph(checkpointer: Any = None) -> Any:
    """Compile and return the RAG subgraph (full pipeline with answer synthesis).

    Used internally by the standalone HTTP server (main.py).
    Provides full Q&A orchestration for direct users or evals.
    """
    cp = checkpointer if checkpointer is not None else MemorySaver()
    return _make_state_graph().compile(checkpointer=cp)


# Alias kept for the standalone FastAPI lifespan which calls build_graph(cp).
build_graph = build_rag_subgraph

# ── CLI / test convenience ────────────────────────────────────────────────────
poc_graph = _make_state_graph().compile(checkpointer=MemorySaver())
