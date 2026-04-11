from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    # Core — always required
    url: str
    title: str
    doc_id: str
    # Optional — present in structured corpora
    section: str | None = None
    language: str = "en"  # ISO 639-1; multilingual-e5-large supports 100+ languages
    parent_id: str | None = None  # parent_doc strategy
    # Corpus-specific fields — None by default, additive when needed
    namespace: str | None = (
        None  # corpus partition, e.g. "docs", "wiki", "api-reference"
    )
    topic: str | None = None  # subject area, e.g. "authentication", "billing", "setup"
    content_type: str | None = None  # "article", "tutorial", "reference", "changelog"
    access_tier: str | None = None  # access control: "public", "internal", "premium"
    last_updated: str | None = None  # ISO 8601, freshness detection
    source_id: str | None = None  # upstream record ID from ingestion source
    completeness_score: float | None = None  # quality gate at ingestion


class Chunk(BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None


class GradedChunk(BaseModel):
    chunk: Chunk
    score: float  # retrieval score
    relevant: bool  # CRAG relevance judgment


class RankedChunk(BaseModel):
    chunk: Chunk
    relevance_score: float = Field(ge=0.0, le=1.0)  # reranker output (0–1)
    rank: int
