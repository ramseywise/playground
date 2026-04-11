"""Pipeline tracing and failure clustering for RAG eval observability.

Contains both the generic pipeline tracing infrastructure and the
RAG-specific extensions (spans, trace wrapper, failure taxonomy).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from typing import Any


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


@dataclass
class SpanInfo:
    """Information about a single pipeline span (step)."""

    name: str
    start_time: datetime
    end_time: datetime | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "success"  # success, error, warning
    error_message: str | None = None


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
# Pipeline Trace
# ---------------------------------------------------------------------------


@dataclass
class PipelineTrace:
    """Complete trace of a pipeline execution."""

    trace_id: str
    query_id: str
    query: str
    timestamp: datetime = field(default_factory=datetime.now)

    spans: list[SpanInfo] = field(default_factory=list)
    answer: str = ""

    confidence: float = 0.0
    status: str = "success"  # success, partial, failure
    failure_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "query_id": self.query_id,
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "total_latency_ms": self.total_latency_ms,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "confidence": self.confidence,
            "n_spans": len(self.spans),
            "answer_length": len(self.answer),
        }


@dataclass
class RAGPipelineTrace:
    """Complete trace of a RAG pipeline execution with per-stage spans."""

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
        return self.generation.total_tokens if self.generation else 0


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class PipelineTracer:
    """Trace pipeline executions."""

    def __init__(self) -> None:
        self.traces: list[PipelineTrace] = []

    def create_trace(self, query_id: str, query: str) -> PipelineTrace:
        trace_id = sha256(
            f"{query_id}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        trace = PipelineTrace(trace_id=trace_id, query_id=query_id, query=query)
        self.traces.append(trace)
        return trace

    def get_traces(self, status: str | None = None) -> list[PipelineTrace]:
        if status:
            return [t for t in self.traces if t.status == status]
        return self.traces

    def get_failure_traces(self) -> list[PipelineTrace]:
        return [t for t in self.traces if t.status == "failure"]


# ---------------------------------------------------------------------------
# Failure Clustering
# ---------------------------------------------------------------------------


@dataclass
class FailureCluster:
    """A cluster of similar failures."""

    cluster_id: str
    failure_type: str
    count: int
    examples: list[PipelineTrace]
    common_patterns: list[str]
    suggested_fix: str


class FailureClusterer:
    """RAG-aware failure clusterer with retrieval-specific taxonomy.

    Covers both generic pipeline failures and RAG-specific failure modes
    (retrieval, ranking, generation, grounding, coverage).
    """

    FAILURE_TYPES: dict[str, str] = {
        "low_confidence": "Low confidence score",
        "timeout": "Pipeline timeout",
        "parse_error": "Output parsing failed",
        "empty_result": "No result produced",
        "retrieval_failure": "Wrong documents retrieved - query-document mismatch",
        "ranking_failure": "Right docs found but buried in results",
        "generation_failure": "Right docs, wrong answer - model didn't use context",
        "grounding_failure": "Answer contains unsupported claims (hallucination)",
        "coverage_gap": "Topic not in knowledge base",
        "complexity_failure": "Question too complex for single retrieval",
        "zero_retrieval": "No relevant chunks retrieved",
        "context_noise": "Too much irrelevant context",
        "unknown": "Unknown failure type",
    }

    def __init__(self, embedder: Callable | None = None) -> None:
        self.embedder = embedder
        self.clusters: list[FailureCluster] = []

    def classify_failure(self, trace: PipelineTrace) -> str:
        if trace.status != "failure":
            return "success"

        extra = trace.extra or {}
        num_retrieved = extra.get("num_retrieved", -1)
        retrieval_confidence = extra.get("retrieval_confidence", trace.confidence)

        if num_retrieved == 0:
            return "coverage_gap"
        if not trace.answer:
            return "empty_result"
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
            if "parse" in reason:
                return "parse_error"

        if trace.confidence < 0.3:
            return "low_confidence"

        return "unknown"

    def cluster_failures(self, traces: list[PipelineTrace]) -> list[FailureCluster]:
        failures = [t for t in traces if t.status == "failure"]
        if not failures:
            return []

        type_groups: dict[str, list[PipelineTrace]] = defaultdict(list)
        for trace in failures:
            type_groups[self.classify_failure(trace)].append(trace)

        clusters = []
        for failure_type, group_traces in type_groups.items():
            common_patterns = _find_common_patterns([t.query for t in group_traces])
            clusters.append(
                FailureCluster(
                    cluster_id=f"{failure_type}_{len(group_traces)}",
                    failure_type=failure_type,
                    count=len(group_traces),
                    examples=group_traces[:5],
                    common_patterns=common_patterns,
                    suggested_fix=self._suggest_fix(failure_type),
                )
            )

        self.clusters = sorted(clusters, key=lambda x: x.count, reverse=True)
        return self.clusters

    def _suggest_fix(self, failure_type: str) -> str:
        suggestions = {
            "low_confidence": "Review scoring thresholds and input quality",
            "timeout": "Optimize pipeline, add caching, reduce input size",
            "parse_error": "Improve output format instructions, add retries",
            "empty_result": "Check pipeline connectivity and input validity",
            "retrieval_failure": "Improve embeddings, add query expansion, check synonyms",
            "ranking_failure": "Tune reranker, adjust fusion weights, check relevance scoring",
            "generation_failure": "Improve prompt template, add few-shot examples",
            "grounding_failure": "Strengthen attribution instructions, enable claim verification",
            "coverage_gap": "Expand corpus with missing content, check content pipeline",
            "complexity_failure": "Route to iterative retrieval, implement query decomposition",
            "zero_retrieval": "Expand query terms, check index coverage, lower similarity threshold",
            "context_noise": "Add reranking, reduce top-k, improve chunk quality",
            "unknown": "Review trace details for patterns",
        }
        return suggestions.get(failure_type, "Review trace details")

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_clusters": len(self.clusters),
            "clusters": [
                {
                    "type": c.failure_type,
                    "count": c.count,
                    "patterns": c.common_patterns,
                    "fix": c.suggested_fix,
                }
                for c in self.clusters
            ],
        }


def _find_common_patterns(queries: list[str]) -> list[str]:
    patterns: list[str] = []
    lengths = [len(q.split()) for q in queries]
    avg_length = statistics.mean(lengths) if lengths else 0

    if avg_length > 20:
        patterns.append("Long/complex queries")
    elif avg_length < 5:
        patterns.append("Very short queries")

    all_words = " ".join(queries).lower().split()
    word_counts: dict[str, int] = defaultdict(int)
    for word in all_words:
        if len(word) > 4:
            word_counts[word] += 1

    total = len(queries)
    for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        if count / total > 0.3:
            patterns.append(f"Contains '{word}'")

    return patterns[:5]
