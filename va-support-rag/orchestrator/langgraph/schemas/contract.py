"""Structured LLM outputs and Q&A contract types (graph state mirrors these)."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from rag.schemas.chunks import GradedChunk, RankedChunk

QAOutcome = Literal["answer", "escalate"]

# Locale codes for query transformation and answer language (BCP-47 subtags).
SUPPORTED_LOCALES: dict[str, str] = {
    "da": "Danish",
    "de": "German",
    "en": "English",
    "fr": "French",
}


def locale_to_language(locale: str | None) -> str:
    """Full language name for a locale tag; falls back to English."""
    return SUPPORTED_LOCALES.get((locale or "").lower().split("-")[0], "English")


class Citation(BaseModel):
    """One retrieved passage cited in an answer."""

    chunk_id: str = ""
    title: str = ""
    url: str = ""
    score: float = 0.0


class LatencyBreakdown(BaseModel):
    """Per-stage latency for observability (milliseconds)."""

    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    policy_retrieval_ms: float = 0.0
    policy_rerank_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0


class QAContextMeta(BaseModel):
    """Optional request context (market / locale)."""

    market: str | None = Field(default=None, description="e.g. DK")
    locale: str | None = Field(default=None, description="e.g. da, en")


class PlannerOutput(BaseModel):
    mode: Literal["q&a", "task_execution"] = Field(
        description='Either "q&a" or "task_execution"',
    )
    intent: Optional[str] = Field(
        default=None,
        description="Short intent label for logging/eval (e.g. billing_question, schedule_task).",
    )
    retrieval_hints: List[str] = Field(
        default_factory=list,
        description="Optional phrases to bias retrieval when mode is q&a (may be ignored by keyword path).",
    )


class ClarifyOutput(BaseModel):
    collected_fields: Optional[Dict[str, str]] = Field(default=None)
    missing_fields: Optional[List[str]] = Field(default=None)


class SchedulerOutput(BaseModel):
    action_steps: List[str] = Field(description="Ordered list of action steps")


class HybridRetrievalProbeOutput(BaseModel):
    """LLM probe for borderline ensemble scores (hybrid policy only)."""

    proceed_to_rerank: bool = Field(
        description="True if reranking may still recover useful evidence for the query.",
    )
    rationale: str = ""


class HybridRerankProbeOutput(BaseModel):
    """LLM probe for borderline rerank confidence (hybrid policy only)."""

    answer_anyway: bool = Field(
        description="True if the passages are sufficient to answer despite low score.",
    )
    rationale: str = ""


class PostAnswerEvalOutput(BaseModel):
    """Structured post-answer check when RAG_POST_ANSWER_EVALUATOR is enabled."""

    verdict: Literal["accept", "escalate", "refine"] = Field(
        description="accept: ship answer; escalate: handoff; refine: retry retrieval",
    )
    public_message: str = Field(
        default="",
        description="If escalate, short user-facing reason (optional).",
    )
    refinement_query: Optional[str] = Field(
        default=None,
        description="If refine, optional replacement query for the retriever.",
    )


class RetrievalQueryTransformOutput(BaseModel):
    """Locale-specific search queries for embedding retrieval (multi-query fusion)."""

    queries: List[str] = Field(
        min_length=2,
        max_length=3,
        description=(
            "Two or three short search queries in the target language: translate the "
            "user's need, plus optional synonyms or rephrasings for better recall."
        ),
    )


def format_graded_context(graded: list[GradedChunk]) -> str:
    if not graded:
        return "No relevant documents found."
    lines: list[str] = []
    for idx, gc in enumerate(graded, start=1):
        m = gc.chunk.metadata
        src = m.title or m.url or m.doc_id or gc.chunk.id
        lines.append(
            f"Document {idx}:\n"
            f"Score: {gc.score:.4f}\n"
            f"Source: {src}\n"
            f"Content:\n{gc.chunk.text}\n"
        )
    return "\n" + "=" * 60 + "\n" + "\n".join(lines)


def format_reranked_context(ranked: list[RankedChunk]) -> str:
    if not ranked:
        return "No relevant documents found."
    lines: list[str] = []
    for rc in ranked:
        m = rc.chunk.metadata
        src = m.title or m.url or m.doc_id or rc.chunk.id
        lines.append(
            f"Rank {rc.rank} (relevance {rc.relevance_score:.4f}):\n"
            f"Source: {src}\n"
            f"Content:\n{rc.chunk.text}\n"
        )
    return "\n" + "=" * 60 + "\n" + "\n".join(lines)


def citations_from_ranked(ranked: list[RankedChunk]) -> list[Citation]:
    out: list[Citation] = []
    for rc in ranked:
        m = rc.chunk.metadata
        out.append(
            Citation(
                chunk_id=rc.chunk.id,
                title=m.title,
                url=m.url,
                score=rc.relevance_score,
            )
        )
    return out


def citations_from_ranked_ordered(
    ranked: list[RankedChunk], chunk_ids: tuple[str, ...]
) -> list[Citation]:
    """Citations aligned to *chunk_ids* order (e.g. after context builder trimming)."""
    by_id = {rc.chunk.id: rc for rc in ranked}
    out: list[Citation] = []
    for cid in chunk_ids:
        rc = by_id.get(cid)
        if rc is None:
            continue
        m = rc.chunk.metadata
        out.append(
            Citation(
                chunk_id=rc.chunk.id,
                title=m.title,
                url=m.url,
                score=rc.relevance_score,
            )
        )
    return out


__all__ = [
    "Citation",
    "ClarifyOutput",
    "HybridRetrievalProbeOutput",
    "HybridRerankProbeOutput",
    "LatencyBreakdown",
    "PlannerOutput",
    "PostAnswerEvalOutput",
    "QAContextMeta",
    "QAOutcome",
    "RetrievalQueryTransformOutput",
    "SchedulerOutput",
    "SUPPORTED_LOCALES",
    "citations_from_ranked",
    "citations_from_ranked_ordered",
    "format_graded_context",
    "format_reranked_context",
    "locale_to_language",
]
