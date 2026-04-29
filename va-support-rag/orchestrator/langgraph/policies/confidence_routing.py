"""Confidence routing: threshold-based answer / gate / escalate from ensemble and rerank scores.

Pure decision logic (unit-testable). Graph nodes and hybrid probes call into this module.

Stable LangGraph **node ids** remain in :mod:`app.graph.graph`; this module is routing logic only.
"""

from __future__ import annotations

from typing import Final, Literal

from rag.schemas.chunks import GradedChunk, RankedChunk

RetrievalPolicyRoute = Literal["rerank", "gate", "escalate"]
RerankPolicyRoute = Literal["answer", "gate", "escalate"]

# Stable reason strings for logs and dashboards (values returned by decide_*).
REASON_ERROR: Final = "error"
REASON_NO_RETRIEVAL_RESULTS: Final = "no_retrieval_results"
REASON_LOW_RETRIEVAL_SCORES: Final = "low_retrieval_scores"
REASON_NO_RERANK_RESULTS: Final = "no_rerank_results"
REASON_LOW_CONFIDENCE: Final = "low_confidence"


def retrieval_signal(graded_chunks: list[GradedChunk]) -> float:
    """Best ensemble (fusion) score from graded chunks."""
    if not graded_chunks:
        return 0.0
    return max(c.score for c in graded_chunks)


def decide_after_retrieval(
    *,
    error: str | None,
    graded_chunks: list[GradedChunk],
    ensemble_threshold: float,
) -> tuple[RetrievalPolicyRoute, str | None]:
    """After candidate retrieval + fusion — strong scores go straight to rerank; weak use HITL gate; hard fail escalates."""
    if error:
        return "escalate", error
    if not graded_chunks:
        return "escalate", REASON_NO_RETRIEVAL_RESULTS
    if retrieval_signal(graded_chunks) >= ensemble_threshold:
        return "rerank", None
    return "gate", REASON_LOW_RETRIEVAL_SCORES


def decide_after_rerank(
    *,
    error: str | None,
    reranked_chunks: list[RankedChunk],
    confidence_score: float,
    threshold: float,
) -> tuple[RerankPolicyRoute, str | None]:
    """After reranking — confident path answers; marginal path uses HITL gate; hard errors escalate."""
    if error:
        return "escalate", error
    if not reranked_chunks:
        return "gate", REASON_NO_RERANK_RESULTS
    if confidence_score < threshold:
        return "gate", REASON_LOW_CONFIDENCE
    return "answer", None


# Back-compat alias for tests / external imports
def decide_qa_branch(
    *,
    error: str | None,
    reranked_chunks: list[RankedChunk],
    confidence_score: float,
    threshold: float,
) -> tuple[Literal["answer", "escalate"], str | None]:
    """Legacy single-stage decision (maps rerank policy to answer|escalate only)."""
    route, reason = decide_after_rerank(
        error=error,
        reranked_chunks=reranked_chunks,
        confidence_score=confidence_score,
        threshold=threshold,
    )
    if route == "answer":
        return "answer", None
    if route == "escalate":
        return "escalate", reason
    # gate → legacy escalate (stable reason strings for callers / tests)
    if reason == REASON_NO_RERANK_RESULTS:
        return "escalate", REASON_NO_RETRIEVAL_RESULTS
    return "escalate", reason or REASON_LOW_CONFIDENCE


__all__ = [
    "REASON_ERROR",
    "REASON_LOW_CONFIDENCE",
    "REASON_LOW_RETRIEVAL_SCORES",
    "REASON_NO_RERANK_RESULTS",
    "REASON_NO_RETRIEVAL_RESULTS",
    "decide_after_retrieval",
    "decide_after_rerank",
    "decide_qa_branch",
    "retrieval_signal",
    "RetrievalPolicyRoute",
    "RerankPolicyRoute",
]
