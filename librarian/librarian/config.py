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
    embedding_model_revision: str = (
        ""  # HuggingFace commit SHA — pin for reproducibility
    )
    embedding_provider: str = "multilingual"  # multilingual | minilm

    # Strategies
    ingestion_strategy: str = "html_aware"
    # chroma | opensearch | duckdb | inmemory
    # NOTE: Chroma uses PersistentClient with a single-writer lock.
    # For multi-worker ingest (parallel Fargate tasks), use opensearch.
    retrieval_strategy: str = "chroma"
    reranker_strategy: str = "cross_encoder"
    planning_mode: Literal["rule_based", "llm"] = "rule_based"

    # Thresholds & retrieval tuning
    confidence_threshold: float = 0.4
    relevance_threshold: float = 0.1  # minimum chunk score to keep
    retrieval_k: int = 10
    reranker_top_k: int = 3
    max_query_variants: int = 3
    max_crag_retries: int = 1
    cache_enabled: bool = True
    cache_max_size: int = 256
    cache_ttl_seconds: int = 300
    rrf_k: int = 60  # RRF smoothing constant (Cormack et al. 2009)
    bm25_weight: float = 0.3  # hybrid search BM25 blend
    vector_weight: float = 0.7  # hybrid search vector blend
    trace_lookup_k: int = 5  # golden trace lookup count
    confirm_expensive_ops: bool = (
        False  # cost gate for generate_synthetic + answer_eval
    )

    # API settings
    api_key: str = ""  # when non-empty, all non-health endpoints require X-API-Key header
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8501"]
    # Override via API_CORS_ORIGINS env var for staging/prod (comma-separated or JSON list)
    api_timeout_seconds: float = 30.0
    api_stream_timeout_seconds: float = 120.0

    # S3 data lake
    s3_bucket: str = ""
    s3_raw_prefix: str = "raw/"
    s3_artifacts_prefix: str = "artifacts/"
    s3_region: str = ""  # falls back to AWS_DEFAULT_REGION / boto3 chain

    # Snowflake (MCP server)
    snowflake_account: str = ""
    snowflake_user: str = ""
    snowflake_password: str = ""
    snowflake_warehouse: str = ""
    snowflake_database: str = ""
    snowflake_schema: str = "PUBLIC"

    # Lambda
    lambda_execution: bool = False

    # LangFuse
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_dataset_name: str = "golden_eval"

    # Eval
    eval_dataset_path: str = (
        ""  # path to golden samples JSONL (external, never committed)
    )

    # Google RAG (Vertex AI Search / Gemini grounding — alternative RAG backend)
    google_project_id: str = ""
    google_location: str = "global"
    google_datastore_id: str = ""  # Vertex AI Search datastore for grounding

    # Bedrock Knowledge Bases (alternative RAG backend for A/B comparison)
    bedrock_knowledge_base_id: str = ""
    bedrock_model_arn: str = ""  # full ARN, e.g. arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-haiku-20241022-v1:0
    bedrock_region: str = ""  # falls back to s3_region / AWS_DEFAULT_REGION

    # Checkpointer — persistent conversation state across task restarts
    checkpoint_backend: str = "memory"  # memory | sqlite | postgres
    checkpoint_sqlite_path: str = ".duckdb/checkpoints.sqlite"
    checkpoint_postgres_url: str = ""  # required when checkpoint_backend=postgres

    # OpenTelemetry (optional — requires the otel extra)
    otel_enabled: bool = False
    otel_exporter: str = "otlp"  # otlp | phoenix
    otel_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "librarian"


settings = LibrarySettings()
