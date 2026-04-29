"""Shared I/O contract for all AgentRuntime implementations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class AgentInput(BaseModel):
    query: str
    thread_id: str
    locale: str | None = None
    market: str | None = None
    metadata: dict[str, Any] = {}


class AgentOutput(BaseModel):
    answer: str
    citations: list[dict] = []
    mode: str | None = None
    latency_ms: dict[str, float] = {}
    escalated: bool = False
    #: True when the run failed with an unexpected exception (vs policy escalation).
    pipeline_error: bool = False


class RetrievalOutput(BaseModel):
    """Structured retrieval results for callers (VA agents) to synthesize answers."""

    query: str
    retrieval_queries: list[str] = []
    documents: list[dict[str, Any]] = []
    confidence_score: float = 0.0
    escalated: bool = False
    escalation_reason: str | None = None
    latency_ms: dict[str, float] = {}
    pipeline_error: bool = False


class StreamEvent(BaseModel):
    kind: Literal["node_start", "node_end", "token", "interrupt", "done", "error"]
    node: str | None = None
    data: dict[str, Any] = {}


class ResumeInput(BaseModel):
    thread_id: str
    value: Any


__all__ = ["AgentInput", "AgentOutput", "RetrievalOutput", "ResumeInput", "StreamEvent"]
