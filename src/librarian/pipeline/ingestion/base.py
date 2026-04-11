"""Ingestion-layer protocols — document splitting and chunking contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agents.librarian.pipeline.schemas.chunks import Chunk


class ChunkerConfig(BaseModel):
    max_tokens: int = 512
    overlap_tokens: int = 64
    min_tokens: int = 50


@runtime_checkable
class Chunker(Protocol):
    def chunk_document(self, doc: dict) -> list[Chunk]: ...
