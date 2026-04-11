"""RAG pipeline tracing and failure clustering.

Extends the generic ``eval.tasks.tracing`` with RAG-specific span types
(RetrievalSpan, GenerationSpan) and a RAG-aware failure taxonomy.

Generic base classes are re-exported for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eval.tasks.tracing import (  # noqa: F401 — re-export for backward compat
    FailureCluster,
    PipelineTrace,
    PipelineTracer,
    SpanInfo,
)
from eval.tasks.tracing import FailureClusterer as _BaseClusterer


# ---------------------------------------------------------------------------
# RAG-specific spans
# ---------------------------------------------------------------------------


@dataclass
class RetrievalSpan(SpanInfo):
    """Retrieval step trace."""

    query: str = ""
    num_retrieved: int = 0
    top_scores: list[float] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    retrieval_method: str = ""  # dense, sparse, hybrid


@dataclass
class GenerationSpan(SpanInfo):
    """Generation step trace."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    temperature: float = 0.0


# ---------------------------------------------------------------------------
# RAG-specific pipeline trace (backward-compatible dataclass)
# ---------------------------------------------------------------------------


@dataclass
class RAGPipelineTrace:
    """Complete trace of a RAG pipeline execution.

    Wraps a generic ``PipelineTrace`` and adds RAG-specific fields.
    """

    trace: PipelineTrace
    retrieval: RetrievalSpan | None = None
    reranking: SpanInfo | None = None
    generation: GenerationSpan | None = None
    retrieved_chunks: list[dict] = field(default_factory=list)
    chunk_attributions: dict[str, list[str]] = field(default_factory=dict)
    retrieval_confidence: float = 0.0
    generation_confidence: float = 0.0

    @property
    def total_latency_ms(self) -> float:
        total = 0.0
        if self.retrieval:
            total += self.retrieval.latency_ms
        if self.reranking:
            total += self.reranking.latency_ms
        if self.generation:
            total += self.generation.latency_ms
        return total

    @property
    def total_tokens(self) -> int:
        if self.generation:
            return self.generation.total_tokens
        return 0


# ---------------------------------------------------------------------------
# RAG failure clusterer — extends generic with retrieval-specific types
# ---------------------------------------------------------------------------


class FailureClusterer(_BaseClusterer):
    """RAG-aware failure clusterer with retrieval-specific taxonomy."""

    FAILURE_TYPES: dict[str, str] = {
        **_BaseClusterer.FAILURE_TYPES,
        "retrieval_failure": "Wrong documents retrieved - query-document mismatch",
        "ranking_failure": "Right docs found but buried in results",
        "generation_failure": "Right docs, wrong answer - model didn't use context",
        "grounding_failure": "Answer contains unsupported claims (hallucination)",
        "coverage_gap": "Topic not in knowledge base",
        "complexity_failure": "Question too complex for single retrieval",
        "zero_retrieval": "No relevant chunks retrieved",
        "context_noise": "Too much irrelevant context",
    }

    def classify_failure(self, trace: PipelineTrace) -> str:
        """Classify failure with RAG-specific heuristics."""
        if trace.status != "failure":
            return "success"

        # Check extra data for RAG-specific signals
        extra = trace.extra or {}
        num_retrieved = extra.get("num_retrieved", -1)
        retrieval_confidence = extra.get("retrieval_confidence", trace.confidence)

        if num_retrieved == 0:
            return "coverage_gap"

        if retrieval_confidence < 0.3:
            return "retrieval_failure"

        if retrieval_confidence < 0.5:
            return "ranking_failure"

        if trace.failure_reason:
            reason = trace.failure_reason.lower()
            if "timeout" in reason:
                return "timeout"
            if "hallucin" in reason or "unsupported" in reason:
                return "grounding_failure"
            if "complex" in reason or "multi-hop" in reason:
                return "complexity_failure"
            if "incomplete" in reason:
                return "generation_failure"
            if "noise" in reason:
                return "context_noise"

        return super().classify_failure(trace)

    def _suggest_fix(self, failure_type: str) -> str:
        rag_suggestions: dict[str, str] = {
            "retrieval_failure": "Improve embeddings, add query expansion, check synonyms",
            "ranking_failure": "Tune reranker, adjust fusion weights, check relevance scoring",
            "generation_failure": "Improve prompt template, add few-shot examples",
            "grounding_failure": "Strengthen attribution instructions, enable claim verification",
            "coverage_gap": "Expand corpus with missing content, check content pipeline",
            "complexity_failure": "Route to iterative retrieval, implement query decomposition",
            "zero_retrieval": "Expand query terms, check index coverage, lower similarity threshold",
            "context_noise": "Add reranking, reduce top-k, improve chunk quality",
        }
        return rag_suggestions.get(failure_type, super()._suggest_fix(failure_type))
