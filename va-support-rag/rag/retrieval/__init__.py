"""RAG retrieval strategies — ensemble, RRF, scoring, snippet search, cache, and pipeline."""

from __future__ import annotations

from rag.retrieval.cache import RetrievalCache
from rag.retrieval.ensemble import EnsembleRetriever
from rag.retrieval.rrf import fuse_rankings
from rag.retrieval.scoring import cosine_similarity, term_overlap
from rag.retrieval.snippet import MetadataDB, SnippetDB, SnippetRetriever

__all__ = [
    "EnsembleRetriever",
    "MetadataDB",
    "RetrievalCache",
    "SnippetDB",
    "SnippetRetriever",
    "cosine_similarity",
    "fuse_rankings",
    "term_overlap",
]
