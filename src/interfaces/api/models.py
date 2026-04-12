from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    """Inbound chat request."""

    query: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    conversation_id: str | None = None  # legacy alias — prefer session_id
    backend: Literal["librarian", "bedrock", "google_adk"] = "librarian"


class ChatResponse(BaseModel):
    """Full (non-streaming) chat response."""

    response: str
    citations: list[dict[str, str]]
    confidence_score: float
    confident: bool = True
    escalate: bool = False
    intent: str
    trace_id: str = ""
    backend: str = "librarian"


class StreamEvent(BaseModel):
    """Single SSE event payload."""

    event: Literal["status", "token", "done", "error"]
    data: Any


class ErrorResponse(BaseModel):
    """Structured error body returned on 4xx/5xx responses."""

    error: str
    trace_id: str = ""
    agent: str = "librarian"
    detail: str = ""


class IngestRequest(BaseModel):
    """Ingestion trigger request — provide exactly one source."""

    s3_key: str | None = Field(None, description="S3 object key to ingest")
    s3_prefix: str | None = Field(None, description="S3 prefix for batch ingestion")
    document: dict[str, str] | None = Field(None, description="Inline document dict")

    @model_validator(mode="after")
    def _require_one_source(self) -> IngestRequest:
        if not any([self.s3_key, self.s3_prefix, self.document]):
            msg = "At least one of s3_key, s3_prefix, or document is required"
            raise ValueError(msg)
        return self


class IngestResultItem(BaseModel):
    doc_id: str
    chunk_count: int
    snippet_count: int
    skipped: bool


class IngestResponse(BaseModel):
    results: list[IngestResultItem]
