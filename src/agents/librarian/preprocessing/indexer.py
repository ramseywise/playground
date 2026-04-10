# Backward-compat shim — canonical location: preprocessing/indexing/indexer.py
from agents.librarian.preprocessing.indexing.indexer import (
    ChunkIndexer,
    build_indexer_for_source,
)

__all__ = ["ChunkIndexer", "build_indexer_for_source"]
