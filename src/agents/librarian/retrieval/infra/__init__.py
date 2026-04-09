"""Retrieval infrastructure — vector store and search engine backends.

Each backend implements the Retriever Protocol from retrieval/base.py.
"""

from agents.librarian.retrieval.infra.chroma import ChromaRetriever
from agents.librarian.retrieval.infra.duckdb import DuckDBRetriever
from agents.librarian.retrieval.infra.inmemory import InMemoryRetriever
from agents.librarian.retrieval.infra.opensearch import OpenSearchRetriever

__all__ = [
    "ChromaRetriever",
    "DuckDBRetriever",
    "InMemoryRetriever",
    "OpenSearchRetriever",
]
