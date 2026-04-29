"""Single import surface for preprocessing: corpus steps, loaders (incl. S3), vector ingestion.

Use:
    from rag.preprocessing.pipeline import (
        preprocess_corpus,
        IngestionPipeline,
        build_ingestion_pipeline,
    )

Implementation modules:
    - ``parsing.corpus`` — ``preprocess_corpus`` (clean / dedup / enrich)
    - ``loaders`` — local Markdown, directory glob, ``S3DocumentLoader``
    - ``ingestion`` — ``IngestionPipeline`` (chunk → embed → vector + snippet + metadata DBs)

``build_ingestion_pipeline`` wires:

- **Vector store** — from :func:`app.rag.retrieval.runtime.get_local_retriever` (``VECTOR_STORE_BACKEND``:
  ``duckdb`` | ``opensearch`` | ``memory``).
- **Metadata + snippets** — always local DuckDB tables in ``rag_index.duckdb`` (``ingest_documents``,
  ``ingest_snippets``) via :func:`app.rag.retrieval.runtime.get_duckdb_index_path`, so prod can use
  OpenSearch for embeddings while keeping idempotent checksum / snippet sidecars on disk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.preprocessing.ingestion import IngestionPipeline, IngestionResult
from rag.preprocessing.faq_csv import load_faq_csv_documents
from rag.preprocessing.loaders import (
    S3DocumentLoader,
    load_directory,
    load_markdown_file,
    parse_frontmatter,
)
from rag.preprocessing.parsing.corpus import preprocess_corpus

if TYPE_CHECKING:
    from rag.preprocessing.base import Chunker


def build_ingestion_pipeline(
    chunker: Chunker | None = None,
    batch_size: int = 64,
) -> IngestionPipeline:
    """Return an :class:`IngestionPipeline` aligned with runtime ``VECTOR_STORE_BACKEND``.

    Vectors are written via :func:`~app.rag.retrieval.runtime.get_local_retriever` (DuckDB file,
    OpenSearch kNN index, or in-memory dict). Document metadata and sentence snippets are always
    persisted to local DuckDB tables alongside the default DuckDB vector path so ingestion stays
    idempotent (checksum dedup) regardless of where chunk embeddings are stored.
    """
    from rag.preprocessing.base import ChunkerConfig
    from rag.preprocessing.chunking.strategies import FixedChunker
    from rag.datastore import DuckDBDocumentMetadataStore, DuckDBSnippetStore
    from rag.embedding import get_embedder_for_indexing
    from rag.datastore.factory import get_duckdb_index_path, get_local_retriever

    path = get_duckdb_index_path()
    ch = chunker or FixedChunker(config=ChunkerConfig())
    return IngestionPipeline(
        chunker=ch,
        embedder=get_embedder_for_indexing(),
        vector_store=get_local_retriever(),
        metadata_db=DuckDBDocumentMetadataStore(path),
        snippet_db=DuckDBSnippetStore(path),
        batch_size=batch_size,
    )


__all__ = [
    "IngestionPipeline",
    "IngestionResult",
    "S3DocumentLoader",
    "build_ingestion_pipeline",
    "load_directory",
    "load_faq_csv_documents",
    "load_markdown_file",
    "parse_frontmatter",
    "preprocess_corpus",
]
