"""Shared component builders — config-driven, framework-agnostic.

All component construction is centralised here.  Strategy selection follows
``LibrarySettings``.  Both LangGraph and ADK factories compose from these
builders, ensuring consistent component creation regardless of orchestration
framework.

Each builder uses lazy imports to avoid pulling in heavy dependencies
(sentence-transformers, chromadb, opensearchpy, etc.) at module load time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from librarian.config import LibrarySettings, settings as _default_settings
from core.logging import get_logger

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    from clients.llm import LLMClient
    from librarian.ingestion.base import Chunker
    from librarian.retrieval.base import Embedder, Retriever
    from librarian.reranker.base import Reranker
    from storage.metadata_db import MetadataDB
    from storage.snippet_db import SnippetDB

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def build_llm(cfg: LibrarySettings) -> LLMClient:
    """Build the primary LLM client (Sonnet-class for generation)."""
    if cfg.llm_provider == "gemini":
        from clients.llm import GeminiLLM

        return GeminiLLM(model=cfg.model_gemini, api_key=cfg.gemini_api_key)

    from clients.llm import AnthropicLLM

    return AnthropicLLM(
        model=cfg.anthropic_model_sonnet,
        api_key=cfg.anthropic_api_key,
    )


def build_history_llm(cfg: LibrarySettings) -> LLMClient:
    """Build a lightweight LLM for conversation condensing (Haiku-class)."""
    if cfg.llm_provider == "gemini":
        from clients.llm import GeminiLLM

        return GeminiLLM(model=cfg.model_gemini, api_key=cfg.gemini_api_key)

    from clients.llm import AnthropicLLM

    return AnthropicLLM(
        model=cfg.anthropic_model_haiku,
        api_key=cfg.anthropic_api_key,
    )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def build_embedder(cfg: LibrarySettings) -> Embedder:
    """Build an embedder based on ``cfg.embedding_provider``."""
    if cfg.embedding_provider == "minilm":
        from librarian.ingestion.embeddings.embedders import MiniLMEmbedder

        return MiniLMEmbedder(model_name=cfg.embedding_model)

    # Default: multilingual (intfloat/multilingual-e5-large)
    from librarian.ingestion.embeddings.embedders import MultilingualEmbedder

    return MultilingualEmbedder(model_name=cfg.embedding_model)


def warm_up_embedder(cfg: LibrarySettings | None = None) -> None:
    """Pre-load the embedding model so the first real request is fast.

    Call once during application startup (e.g. in the FastAPI lifespan).
    The model cache is process-wide, so any embedder instance created later
    will find the model already loaded.
    """
    cfg = cfg or _default_settings
    embedder = build_embedder(cfg)
    embedder.embed_query("warmup")
    log.info("components.embedder_warmup.done", model=cfg.embedding_model)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def build_retriever(cfg: LibrarySettings, embedder: Embedder) -> Retriever:
    """Build a retriever based on ``cfg.retrieval_strategy``."""
    if cfg.retrieval_strategy == "inmemory":
        from storage.vectordb.inmemory import InMemoryRetriever

        return InMemoryRetriever()

    if cfg.retrieval_strategy == "opensearch":
        from storage.vectordb.opensearch import OpenSearchRetriever

        return OpenSearchRetriever(
            index=cfg.opensearch_index,
            bm25_weight=cfg.bm25_weight,
            vector_weight=cfg.vector_weight,
        )

    if cfg.retrieval_strategy == "duckdb":
        from storage.vectordb.duckdb import DuckDBRetriever

        return DuckDBRetriever(
            db_path=cfg.duckdb_path,
            bm25_weight=cfg.bm25_weight,
            vector_weight=cfg.vector_weight,
        )

    # Default: chroma (persistent, no Docker required)
    from storage.vectordb.chroma import ChromaRetriever

    return ChromaRetriever(
        persist_dir=cfg.chroma_persist_dir,
        collection_name=cfg.chroma_collection,
        bm25_weight=cfg.bm25_weight,
        vector_weight=cfg.vector_weight,
    )


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


def build_reranker(cfg: LibrarySettings, llm: LLMClient) -> Reranker:
    """Build a reranker based on ``cfg.reranker_strategy``."""
    if cfg.reranker_strategy == "llm_listwise":
        from librarian.reranker.llm_listwise import LLMListwiseReranker

        return LLMListwiseReranker(llm=llm)

    if cfg.reranker_strategy == "passthrough":
        from librarian.reranker.passthrough import PassthroughReranker

        return PassthroughReranker()

    # Default: cross_encoder
    from librarian.reranker.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker()


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def build_chunker(cfg: LibrarySettings) -> Chunker:
    """Build a chunker based on ``cfg.ingestion_strategy``."""
    from librarian.ingestion.chunking.strategies import (
        AdjacencyChunker,
        FixedChunker,
        OverlappingChunker,
        StructuredChunker,
    )
    from librarian.ingestion.chunking.html_aware import HtmlAwareChunker
    from librarian.ingestion.chunking.parent_doc import ParentDocChunker

    dispatch = {
        "fixed": FixedChunker,
        "overlapping": OverlappingChunker,
        "structured": StructuredChunker,
        "adjacency": AdjacencyChunker,
        "parent_doc": ParentDocChunker,
        "html_aware": HtmlAwareChunker,
    }
    cls = dispatch.get(cfg.ingestion_strategy, HtmlAwareChunker)
    return cls()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def build_storage(cfg: LibrarySettings) -> tuple[MetadataDB, SnippetDB]:
    """Build ``(MetadataDB, SnippetDB)`` backed by ``cfg.duckdb_path``."""
    from pathlib import Path

    from storage.metadata_db import MetadataDB
    from storage.snippet_db import SnippetDB

    db_path = cfg.duckdb_path
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    metadata_db = MetadataDB(db_path)
    snippet_db = SnippetDB(db_path)
    return cast(tuple[MetadataDB, SnippetDB], (metadata_db, snippet_db))


# ---------------------------------------------------------------------------
# Checkpointer (LangGraph-specific but framework-agnostic build logic)
# ---------------------------------------------------------------------------


def build_checkpointer(cfg: LibrarySettings) -> BaseCheckpointSaver | None:
    """Build a LangGraph checkpointer based on ``cfg.checkpoint_backend``.

    - ``memory``  — in-process ``MemorySaver`` (default, no persistence)
    - ``sqlite``  — ``SqliteSaver`` backed by ``cfg.checkpoint_sqlite_path``
    - ``postgres`` — ``PostgresSaver`` via ``cfg.checkpoint_postgres_url``

    Returns ``None`` only when the backend is explicitly unknown.
    """
    from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: F811

    backend = cfg.checkpoint_backend

    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        log.info("components.checkpointer", backend="memory")
        return MemorySaver()

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            msg = (
                "langgraph-checkpoint-sqlite is required for checkpoint_backend='sqlite'. "
                "Install with: uv add langgraph-checkpoint-sqlite"
            )
            raise ImportError(msg) from exc

        from pathlib import Path

        Path(cfg.checkpoint_sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        log.info(
            "components.checkpointer",
            backend="sqlite",
            path=cfg.checkpoint_sqlite_path,
        )
        return SqliteSaver.from_conn_string(cfg.checkpoint_sqlite_path)

    if backend == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            msg = (
                "langgraph-checkpoint-postgres is required for checkpoint_backend='postgres'. "
                "Install with: uv add langgraph-checkpoint-postgres"
            )
            raise ImportError(msg) from exc

        if not cfg.checkpoint_postgres_url:
            msg = (
                "CHECKPOINT_POSTGRES_URL must be set when checkpoint_backend='postgres'"
            )
            raise ValueError(msg)

        log.info("components.checkpointer", backend="postgres")
        return PostgresSaver.from_conn_string(cfg.checkpoint_postgres_url)

    log.warning("components.checkpointer.unknown", backend=backend)
    return None
