"""Vector indexes, DuckDB sidecar tables, and the runtime factory."""

from __future__ import annotations

from rag.datastore.duckdb import (
    DuckDBDocumentMetadataStore,
    DuckDBSnippetStore,
)
from rag.datastore.factory import (
    bootstrap_txt_corpus,
    get_duckdb_index_path,
    get_local_retriever,
    get_vectorstore,
    reset_vectorstore_for_tests,
)
from rag.datastore.local import (
    ChromaVectorIndex,
    DictVectorIndex,
    DuckDBVectorIndex,
)
from rag.datastore.opensearch import OpenSearchVectorIndex

__all__ = [
    "ChromaVectorIndex",
    "DictVectorIndex",
    "DuckDBDocumentMetadataStore",
    "DuckDBSnippetStore",
    "DuckDBVectorIndex",
    "OpenSearchVectorIndex",
    "bootstrap_txt_corpus",
    "get_duckdb_index_path",
    "get_local_retriever",
    "get_vectorstore",
    "reset_vectorstore_for_tests",
]
