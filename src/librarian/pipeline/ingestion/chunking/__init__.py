"""Chunking strategies — modularized by method.

- utils.py:      Shared helpers (make_doc_id, splitting primitives, measurement)
- strategies.py: FixedChunker, OverlappingChunker, StructuredChunker, AdjacencyChunker
- html_aware.py: HtmlAwareChunker (heading-boundary + recursive fallback)
- parent_doc.py: ParentDocChunker (two-level parent/child strategy)
"""

from agents.librarian.pipeline.ingestion.chunking.html_aware import HtmlAwareChunker
from agents.librarian.pipeline.ingestion.chunking.parent_doc import ParentDocChunker
from agents.librarian.pipeline.ingestion.chunking.strategies import (
    AdjacencyChunker,
    FixedChunker,
    OverlappingChunker,
    StructuredChunker,
)

__all__ = [
    "AdjacencyChunker",
    "FixedChunker",
    "HtmlAwareChunker",
    "OverlappingChunker",
    "ParentDocChunker",
    "StructuredChunker",
]
