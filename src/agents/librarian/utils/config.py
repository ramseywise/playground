"""Librarian-specific configuration — extends infra.config.BaseSettings."""

from __future__ import annotations

from typing import Literal

from core.config.settings import BaseSettings


class LibrarySettings(BaseSettings):
    """RAG agent settings.  Inherits API keys + model names from BaseSettings."""

    # Aliases for backward compat within librarian code
    @property
    def anthropic_model_haiku(self) -> str:
        return self.model_haiku

    @property
    def anthropic_model_sonnet(self) -> str:
        return self.model_sonnet

    # OpenSearch
    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "librarian-chunks"
    opensearch_user: str = "admin"
    opensearch_password: str = ""

    # ChromaDB
    chroma_persist_dir: str = ".chroma"
    chroma_collection: str = "librarian-chunks"

    # DuckDB — shared across metadata, traces, and graph tables
    duckdb_path: str = ".duckdb/librarian.db"

    # Embedding
    embedding_model: str = "intfloat/multilingual-e5-large"

    # Strategies
    ingestion_strategy: str = "html_aware"
    retrieval_strategy: str = "chroma"  # chroma | opensearch | duckdb | inmemory
    reranker_strategy: str = "cross_encoder"
    planning_mode: Literal["rule_based", "llm"] = "rule_based"

    # Thresholds & retrieval tuning
    confidence_threshold: float = 0.4
    relevance_threshold: float = 0.1  # minimum chunk score to keep
    retrieval_k: int = 10
    reranker_top_k: int = 3
    max_query_variants: int = 3
    max_crag_retries: int = 1
    bm25_weight: float = 0.3  # hybrid search BM25 blend
    vector_weight: float = 0.7  # hybrid search vector blend
    trace_lookup_k: int = 5  # golden trace lookup count
    confirm_expensive_ops: bool = (
        False  # cost gate for generate_synthetic + answer_eval
    )

    # LangFuse
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


settings = LibrarySettings()
