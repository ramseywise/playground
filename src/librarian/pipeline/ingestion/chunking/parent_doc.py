from __future__ import annotations

import uuid

from agents.librarian.pipeline.ingestion.base import ChunkerConfig
from agents.librarian.pipeline.ingestion.chunking.html_aware import HtmlAwareChunker
from agents.librarian.pipeline.ingestion.chunking.utils import (
    recursive_split_by_separators as _recursive_split,
    word_count as _word_count,
)
from agents.librarian.pipeline.schemas.chunks import Chunk, ChunkMetadata


class ParentDocChunker:
    """Two-level chunking strategy.

    Parent chunks (full sections) are stored for generation context.
    Child chunks (small, overlapping) are indexed for retrieval and tagged with ``parent_id``.

    Child chunk IDs follow the pattern ``{parent_id}_child{i}``.
    """

    def __init__(
        self,
        parent_config: ChunkerConfig | None = None,
        child_config: ChunkerConfig | None = None,
    ) -> None:
        # Parents: large windows — not size-filtered for retrieval, kept for generation
        self.parent_config = parent_config or ChunkerConfig(
            max_tokens=512, overlap_tokens=0, min_tokens=1
        )
        # Children: small windows — what actually gets indexed
        self.child_config = child_config or ChunkerConfig(
            max_tokens=128, overlap_tokens=32, min_tokens=20
        )
        self._section_chunker = HtmlAwareChunker(config=self.parent_config)

    def chunk_document(self, doc: dict) -> list[Chunk]:
        """Return child chunks tagged with parent_id. Parents are embedded inline as metadata."""
        parent_chunks = self._section_chunker.chunk_document(doc)
        all_chunks: list[Chunk] = []

        for parent in parent_chunks:
            parent_id = str(uuid.uuid4())
            children = _recursive_split(
                parent.text,
                self.child_config.max_tokens,
                self.child_config.overlap_tokens,
            )
            for i, child_text in enumerate(children):
                if _word_count(child_text) < self.child_config.min_tokens:
                    continue
                child_meta = ChunkMetadata(
                    url=parent.metadata.url,
                    title=parent.metadata.title,
                    doc_id=parent.metadata.doc_id,
                    section=parent.metadata.section,
                    language=parent.metadata.language,
                    parent_id=parent_id,
                    namespace=parent.metadata.namespace,
                    topic=parent.metadata.topic,
                    content_type=parent.metadata.content_type,
                    access_tier=parent.metadata.access_tier,
                    source_id=parent.metadata.source_id,
                )
                all_chunks.append(
                    Chunk(
                        id=f"{parent_id}_child{i}",
                        text=child_text,
                        metadata=child_meta,
                    )
                )

        return all_chunks
