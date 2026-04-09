from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class LibrarySettings(BaseSettings):
    anthropic_api_key: str = ""
    anthropic_model_haiku: str = "claude-haiku-4-5-20251001"
    anthropic_model_sonnet: str = "claude-sonnet-4-6"

    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "librarian-chunks"
    opensearch_user: str = "admin"
    opensearch_password: str = ""

    chroma_persist_dir: str = ".chroma"
    chroma_collection: str = "librarian-chunks"

    duckdb_path: str = ".duckdb/librarian.db"

    # Swap to "intfloat/e5-large-v2" for English-only, "intfloat/e5-small-v2" for lightweight
    embedding_model: str = "intfloat/multilingual-e5-large"

    ingestion_strategy: str = "html_aware"
    retrieval_strategy: str = "chroma"  # chroma | opensearch | duckdb | inmemory
    reranker_strategy: str = "cross_encoder"
    planning_mode: Literal["rule_based", "llm"] = "rule_based"

    confidence_threshold: float = 0.4
    retrieval_k: int = 10
    reranker_top_k: int = 3
    max_query_variants: int = 3
    max_crag_retries: int = 1
    confirm_expensive_ops: bool = (
        False  # cost gate for generate_synthetic + answer_eval
    )

    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = LibrarySettings()
