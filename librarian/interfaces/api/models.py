from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from interfaces.api.backends import BackendLiteral


class ChatRequest(BaseModel):
    """Inbound chat request."""

    query: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    conversation_id: str | None = None  # legacy alias — prefer session_id
    backend: BackendLiteral = "librarian"


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


class ErrorResponse(BaseModel):
    """Structured error body returned on 4xx/5xx responses."""

    error: str
    trace_id: str = ""
    agent: str = "librarian"
    detail: str = ""


class BackendInfo(BaseModel):
    """Describes a single backend's availability and capabilities."""

    id: str
    label: str
    available: bool
    streaming: bool = False


class BackendsResponse(BaseModel):
    """Response from the /backends discovery endpoint."""

    backends: list[BackendInfo]


class InlineDocument(BaseModel):
    """Structured inline document payload for the /ingest endpoint."""

    text: str = Field(..., min_length=1, description="Document body text")
    url: str = Field("", description="Source URL")
    title: str = Field("", description="Document title")
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Ingestion trigger request — provide exactly one source."""

    s3_key: str | None = Field(None, description="S3 object key to ingest", pattern=r"^raw/")
    s3_prefix: str | None = Field(None, description="S3 prefix for batch ingestion", pattern=r"^raw/")
    document: InlineDocument | None = Field(None, description="Inline document")

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
