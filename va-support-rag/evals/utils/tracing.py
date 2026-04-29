"""Lightweight pipeline tracing for failure clustering in eval runs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineTrace:
    task_id: str
    query: str
    status: str = "pending"
    confidence: float = 0.0
    failure_reason: str | None = None


@dataclass
class FailureCluster:
    failure_type: str
    count: int
    common_patterns: list[str] = field(default_factory=list)


class PipelineTracer:
    def __init__(self) -> None:
        self._traces: dict[str, PipelineTrace] = {}

    def create_trace(self, task_id: str, query: str) -> PipelineTrace:
        t = PipelineTrace(task_id=task_id, query=query)
        self._traces[task_id] = t
        return t

    def get_failure_traces(self) -> list[PipelineTrace]:
        return [t for t in self._traces.values() if t.status != "success"]


class FailureClusterer:
    def cluster_failures(self, traces: list[PipelineTrace]) -> list[FailureCluster]:
        if not traces:
            return []
        by_reason: dict[str, list[PipelineTrace]] = {}
        for t in traces:
            by_reason.setdefault(t.failure_reason or "unknown", []).append(t)
        return [
            FailureCluster(
                failure_type=reason,
                count=len(group),
                common_patterns=[t.query[:80] for t in group[:3]],
            )
            for reason, group in sorted(by_reason.items())
        ]
