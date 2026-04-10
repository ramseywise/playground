"""RAG-specific protocol definitions — the system's shape at a glance.

Import any protocol from here:

    from agents.librarian.protocols import Embedder, Retriever, Reranker, Chunker
"""

from __future__ import annotations

from agents.librarian.protocols.chunker import Chunker, ChunkerConfig
from agents.librarian.protocols.embedder import Embedder
from agents.librarian.protocols.reranker import Reranker
from agents.librarian.protocols.retriever import Retriever

__all__ = [
    "Chunker",
    "ChunkerConfig",
    "Embedder",
    "Reranker",
    "Retriever",
]
