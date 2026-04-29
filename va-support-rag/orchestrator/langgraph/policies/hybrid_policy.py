"""Optional hybrid policy: LLM probes on borderline score bands (see RAG_POLICY_MODE)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from orchestrator.langgraph.policies.confidence_routing import (
    REASON_LOW_CONFIDENCE,
    REASON_LOW_RETRIEVAL_SCORES,
    retrieval_signal,
)
from rag.schemas.chunks import GradedChunk, RankedChunk

if TYPE_CHECKING:
    from orchestrator.langgraph.policies.confidence_routing import (
        RetrievalPolicyRoute,
        RerankPolicyRoute,
    )

log = logging.getLogger(__name__)


def is_borderline_retrieval(signal: float, threshold: float, low_frac: float) -> bool:
    """True when signal is below *threshold* but not clearly useless."""
    if threshold <= 0:
        return False
    return threshold * low_frac <= signal < threshold


def is_borderline_rerank(confidence: float, threshold: float, low_frac: float) -> bool:
    """True when confidence is below *threshold* but in the uncertain band."""
    if threshold <= 0:
        return False
    return threshold * low_frac <= confidence < threshold


def maybe_upgrade_retrieval_route(
    *,
    route: "RetrievalPolicyRoute",
    reason: str | None,
    graded_chunks: list[GradedChunk],
    ensemble_threshold: float,
    border_low_frac: float,
    query: str,
) -> tuple["RetrievalPolicyRoute", str | None]:
    """If hybrid mode and borderline low retrieval scores, ask LLM whether to try rerank."""
    from core.config import RAG_POLICY_MODE

    if RAG_POLICY_MODE != "hybrid":
        return route, reason
    if route != "gate" or reason != REASON_LOW_RETRIEVAL_SCORES:
        return route, reason
    sig = retrieval_signal(graded_chunks)
    if not is_borderline_retrieval(sig, ensemble_threshold, border_low_frac):
        return route, reason

    try:
        from clients import llm as llm_mod
        from orchestrator.langgraph.chains import get_hybrid_retrieval_probe_chain

        if not llm_mod.llm_configured():
            log.info("hybrid_retrieval.skip reason=llm_not_configured")
            return route, reason
        titles = []
        for gc in graded_chunks[:3]:
            m = gc.chunk.metadata
            titles.append(m.title or m.url or gc.chunk.id or "?")
        probe = get_hybrid_retrieval_probe_chain().invoke(
            {
                "query": query,
                "retrieval_score": f"{sig:.4f}",
                "threshold": f"{ensemble_threshold:.4f}",
                "doc_hints": "; ".join(titles) or "(no titles)",
            }
        )
        if probe.proceed_to_rerank:
            log.info(
                "hybrid_retrieval.upgrade ensemble_signal=%.4f ensemble_threshold=%.4f",
                sig,
                ensemble_threshold,
            )
            return "rerank", None
    except Exception:
        log.exception("hybrid_retrieval.probe_failed")
    return route, reason


def maybe_upgrade_rerank_route(
    *,
    route: "RerankPolicyRoute",
    reason: str | None,
    reranked_chunks: list[RankedChunk],
    confidence_score: float,
    threshold: float,
    border_low_frac: float,
    query: str,
) -> tuple["RerankPolicyRoute", str | None]:
    """If hybrid mode and borderline rerank confidence, ask LLM whether to answer anyway."""
    from core.config import RAG_POLICY_MODE

    if RAG_POLICY_MODE != "hybrid":
        return route, reason
    if route != "gate" or reason != REASON_LOW_CONFIDENCE:
        return route, reason
    if not is_borderline_rerank(confidence_score, threshold, border_low_frac):
        return route, reason

    try:
        from clients import llm as llm_mod
        from orchestrator.langgraph.chains import get_hybrid_rerank_probe_chain

        if not llm_mod.llm_configured():
            log.info("hybrid_rerank.skip reason=llm_not_configured")
            return route, reason
        snippets = []
        for rc in reranked_chunks[:3]:
            snippets.append(rc.chunk.text[:400].replace("\n", " "))
        probe = get_hybrid_rerank_probe_chain().invoke(
            {
                "query": query,
                "confidence": f"{confidence_score:.4f}",
                "threshold": f"{threshold:.4f}",
                "snippets": "\n---\n".join(snippets) or "(empty)",
            }
        )
        if probe.answer_anyway:
            log.info(
                "hybrid_rerank.upgrade rerank_top_score=%.4f rerank_threshold=%.4f",
                confidence_score,
                threshold,
            )
            return "answer", None
    except Exception:
        log.exception("hybrid_rerank.probe_failed")
    return route, reason


__all__ = [
    "is_borderline_rerank",
    "is_borderline_retrieval",
    "maybe_upgrade_retrieval_route",
    "maybe_upgrade_rerank_route",
]
