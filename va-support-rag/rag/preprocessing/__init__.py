"""RAG preprocessing: corpus normalization, chunking, indexing, ingestion (offline / batch).

See ``README.md`` in this package for how this differs from query-time retrieval in ``app/rag/retrieval/``.
"""

from __future__ import annotations

from rag.preprocessing.pipeline import (
    IngestionPipeline,
    IngestionResult,
    S3DocumentLoader,
    build_ingestion_pipeline,
    load_directory,
    load_markdown_file,
    parse_frontmatter,
    preprocess_corpus,
)

__all__ = [
    "IngestionPipeline",
    "IngestionResult",
    "S3DocumentLoader",
    "build_ingestion_pipeline",
    "load_directory",
    "load_markdown_file",
    "parse_frontmatter",
    "preprocess_corpus",
]
