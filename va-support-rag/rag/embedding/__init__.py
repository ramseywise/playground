"""Embedding model + LangChain bridge — same stack for indexing and query-time retrieval."""

from __future__ import annotations

from typing import Any

from rag.embedding.bridge import LangChainEmbeddingsBridge
from rag.embedding.sentence_transformers import (
    get_embeddings,
    load_sentence_transformer,
)


def get_embedder_for_indexing() -> Any:
    """Return :class:`~src.rag.retrieval.protocols.Embedder` for chunking pipelines (passage prefixes)."""
    return LangChainEmbeddingsBridge(get_embeddings())


__all__ = [
    "LangChainEmbeddingsBridge",
    "get_embedder_for_indexing",
    "get_embeddings",
    "load_sentence_transformer",
]
