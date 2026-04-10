"""Re-export from canonical location: ingestion/indexing/."""

from agents.librarian.ingestion.indexing.indexer import (  # noqa: F401
    ChunkIndexer,
    build_indexer_for_source,
)

__all__ = ["ChunkIndexer", "build_indexer_for_source"]
