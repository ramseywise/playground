"""Chunk data models — core unit of retrieval and reranking."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    url: str = ""
    title: str = ""
    doc_id: str = ""
    topic: str | None = None
    section: str | None = None
    language: str | None = None
    parent_id: str | None = None
    namespace: str | None = None
    access_tier: str | None = None
    source_id: str | None = None
    content_type: str | None = None


class Chunk(BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    embedding: list[float] | None = None


class GradedChunk(BaseModel):
    """Chunk annotated with a retrieval score and a relevance flag."""

    chunk: Chunk
    score: float
    relevant: bool = False


class RankedChunk(BaseModel):
    """Chunk after reranking — has an ordinal rank and a normalised relevance score."""

    chunk: Chunk
    relevance_score: float
    rank: int


__all__ = ["Chunk", "ChunkMetadata", "GradedChunk", "RankedChunk"]
