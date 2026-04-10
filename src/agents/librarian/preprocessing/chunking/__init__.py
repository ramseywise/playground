"""Re-export from canonical location: ingestion/chunking/.

Chunking strategies now live in ``agents.librarian.ingestion.chunking``.
"""

from agents.librarian.ingestion.chunking.html_aware import HtmlAwareChunker  # noqa: F401
from agents.librarian.ingestion.chunking.parent_doc import ParentDocChunker  # noqa: F401
from agents.librarian.ingestion.chunking.strategies import (  # noqa: F401
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
