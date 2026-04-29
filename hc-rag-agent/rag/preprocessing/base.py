"""Ingestion-layer protocols — document splitting and chunking contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from typing import Literal

from pydantic import BaseModel

from rag.schemas.chunks import Chunk


class ChunkerConfig(BaseModel):
    max_tokens: int = 512
    overlap_tokens: int = 64
    min_tokens: int = 50
    #: ``hash`` — legacy :func:`~app.rag.preprocessing.chunking.utils.make_chunk` ids;
    #: ``stable`` — ``{stable_doc_id}_{n}`` (e.g. ``help_10570020_0``) for eval alignment.
    chunk_id_mode: Literal["hash", "stable"] = "hash"


@runtime_checkable
class Chunker(Protocol):
    def chunk_document(self, doc: dict) -> list[Chunk]: ...
