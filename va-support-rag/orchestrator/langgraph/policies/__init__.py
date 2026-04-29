"""Confidence routing and hybrid probes (threshold + optional LLM border checks).

LangGraph **node names** (e.g. ``qa_policy_retrieval``) are defined in :mod:`app.graph.graph`
and are unchanged here so checkpoints and traces stay stable.

Imports may use :mod:`app.graph.policies` or the legacy paths :mod:`app.graph.confidence_routing`
and :mod:`app.graph.hybrid_policy`.
"""

from __future__ import annotations

from orchestrator.langgraph.policies.confidence_routing import (
    REASON_ERROR,
    REASON_LOW_CONFIDENCE,
    REASON_LOW_RETRIEVAL_SCORES,
    REASON_NO_RERANK_RESULTS,
    REASON_NO_RETRIEVAL_RESULTS,
    decide_after_retrieval,
    decide_after_rerank,
    decide_qa_branch,
    retrieval_signal,
    RetrievalPolicyRoute,
    RerankPolicyRoute,
)
from orchestrator.langgraph.policies.hybrid_policy import (
    is_borderline_rerank,
    is_borderline_retrieval,
    maybe_upgrade_retrieval_route,
    maybe_upgrade_rerank_route,
)

__all__ = [
    "REASON_ERROR",
    "REASON_LOW_CONFIDENCE",
    "REASON_LOW_RETRIEVAL_SCORES",
    "REASON_NO_RERANK_RESULTS",
    "REASON_NO_RETRIEVAL_RESULTS",
    "RetrievalPolicyRoute",
    "RerankPolicyRoute",
    "decide_after_retrieval",
    "decide_after_rerank",
    "decide_qa_branch",
    "is_borderline_rerank",
    "is_borderline_retrieval",
    "maybe_upgrade_retrieval_route",
    "maybe_upgrade_rerank_route",
    "retrieval_signal",
]
