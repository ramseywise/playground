"""Pipeline tracing and failure clustering for eval observability.

Generic tracing infrastructure: track pipeline steps, cluster failures,
and build an attribution taxonomy.  Agent-specific span types (e.g.
RetrievalSpan, GenerationSpan) extend ``SpanInfo`` in their own packages.
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

    # Quality signals
    confidence: float = 0.0

    # Outcome
    status: str = "success"  # success, partial, failure
    failure_reason: str | None = None

    # Agent-specific extension data
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
    """Cluster failures by type to identify systematic issues.

    Agents can extend ``FAILURE_TYPES`` and override ``classify_failure``
    for domain-specific taxonomy (e.g. RAG retrieval/generation failures).
    """

    FAILURE_TYPES: dict[str, str] = {
        "low_confidence": "Low confidence score",
        "timeout": "Pipeline timeout",
        "parse_error": "Output parsing failed",
        "empty_result": "No result produced",
        "unknown": "Unknown failure type",
    }

    def __init__(self, embedder: Callable | None = None) -> None:
        self.embedder = embedder
        self.clusters: list[FailureCluster] = []

    def classify_failure(self, trace: PipelineTrace) -> str:
        """Classify a single failure.  Override for domain-specific taxonomy."""
        if trace.status != "failure":
            return "success"

        if not trace.answer:
            return "empty_result"

        if trace.confidence < 0.3:
            return "low_confidence"

        if trace.failure_reason:
            reason = trace.failure_reason.lower()
            if "timeout" in reason:
                return "timeout"
            if "parse" in reason:
                return "parse_error"

        return "unknown"

    def cluster_failures(self, traces: list[PipelineTrace]) -> list[FailureCluster]:
        """Cluster failures by type and similarity."""
        failures = [t for t in traces if t.status == "failure"]
        if not failures:
            return []

        type_groups: dict[str, list[PipelineTrace]] = defaultdict(list)
        for trace in failures:
            failure_type = self.classify_failure(trace)
            type_groups[failure_type].append(trace)

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
    """Find common patterns in a list of queries."""
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
