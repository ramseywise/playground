"""Dependency injection for the FastAPI application."""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from librarian.factory import create_librarian, warm_up_embedder
from orchestration.nodes.generation import GeneratorAgent
from clients.bedrock import BedrockKBClient
from clients.google_vertex import GoogleRAGClient
from librarian.ingestion.pipeline import IngestionPipeline
from librarian.config import LibrarySettings, settings as _default_settings
from interfaces.api.triage import TriageService
from core.llm import AnthropicLLM
from core.logging import get_logger

log = get_logger(__name__)

_graph: CompiledStateGraph | None = None
_generation_sg: GeneratorAgent | None = None
_pipeline: IngestionPipeline | None = None
_bedrock_client: BedrockKBClient | None = None
_google_adk_client: GoogleRAGClient | None = None
_triage: TriageService | None = None
_settings: LibrarySettings = _default_settings


def init_graph(cfg: LibrarySettings | None = None) -> None:
    """Initialise the graph singleton. Called once at app startup."""
    global _graph, _generation_sg, _bedrock_client, _google_adk_client, _settings  # noqa: PLW0603
    _settings = cfg or _default_settings

    log.info("api.deps.init_graph", retrieval=_settings.retrieval_strategy)
    _graph = create_librarian(_settings)

    # Warm up the embedding model so the first request doesn't pay cold-start cost.
    # _MODEL_CACHE is process-wide — the graph's embedder will find the model hot.
    warm_up_embedder(_settings)

    llm = AnthropicLLM(
        model=_settings.anthropic_model_sonnet,
        api_key=_settings.anthropic_api_key,
    )
    _generation_sg = GeneratorAgent(
        llm=llm,
        confidence_threshold=_settings.confidence_threshold,
    )

    # Bedrock KB client — optional, only if configured
    if _settings.bedrock_knowledge_base_id:
        try:
            _bedrock_client = BedrockKBClient(_settings)
            log.info(
                "api.deps.bedrock_kb.init", kb_id=_settings.bedrock_knowledge_base_id
            )
        except Exception:
            log.warning("api.deps.bedrock_kb.init_failed", exc_info=True)
            _bedrock_client = None
    else:
        log.info(
            "api.deps.bedrock_kb.skipped", reason="no knowledge_base_id configured"
        )

    # Google RAG client — optional, only if configured
    if _settings.google_datastore_id or _settings.google_project_id:
        try:
            _google_adk_client = GoogleRAGClient(_settings)
            log.info(
                "api.deps.google_adk.init",
                datastore_id=_settings.google_datastore_id,
                project_id=_settings.google_project_id,
            )
        except Exception:
            log.warning("api.deps.google_adk.init_failed", exc_info=True)
            _google_adk_client = None
    else:
        log.info("api.deps.google_adk.skipped", reason="no google config set")


def get_graph() -> CompiledStateGraph:
    """Return the compiled LangGraph graph."""
    if _graph is None:
        msg = "Graph not initialised — call init_graph() first"
        raise RuntimeError(msg)
    return _graph


def get_generator_agent() -> GeneratorAgent:
    """Return the generator agent for streaming."""
    if _generation_sg is None:
        msg = "Generator agent not initialised — call init_graph() first"
        raise RuntimeError(msg)
    return _generation_sg


# Backward-compatible alias
get_generation_subgraph = get_generator_agent


def init_pipeline(cfg: LibrarySettings | None = None) -> None:
    """Initialise the ingestion pipeline singleton."""
    global _pipeline  # noqa: PLW0603
    from librarian.factory import create_ingestion_pipeline

    _pipeline = create_ingestion_pipeline(cfg or _settings)
    log.info("api.deps.init_pipeline")


def get_pipeline() -> IngestionPipeline:
    """Return the ingestion pipeline."""
    if _pipeline is None:
        msg = "Pipeline not initialised — call init_pipeline() first"
        raise RuntimeError(msg)
    return _pipeline


def get_bedrock_client() -> BedrockKBClient | None:
    """Return the Bedrock KB client, or None if not configured."""
    return _bedrock_client


def get_google_adk_client() -> GoogleRAGClient | None:
    """Return the Google RAG client, or None if not configured."""
    return _google_adk_client


def is_graph_ready() -> bool:
    """Return True when the librarian graph singleton is initialised."""
    return _graph is not None


def is_bedrock_available() -> bool:
    """Return True when a Bedrock KB client is configured."""
    return _bedrock_client is not None


def is_google_adk_available() -> bool:
    """Return True when a Google RAG client is configured."""
    return _google_adk_client is not None


def init_triage() -> None:
    """Initialise the triage singleton. Called once at app startup."""
    global _triage  # noqa: PLW0603
    _triage = TriageService(
        graph_ready=is_graph_ready,
        bedrock_available=is_bedrock_available,
        google_adk_available=is_google_adk_available,
    )
    log.info("api.deps.init_triage")


def get_triage() -> TriageService:
    """Return the triage service."""
    if _triage is None:
        msg = "Triage not initialised — call init_triage() first"
        raise RuntimeError(msg)
    return _triage


def get_settings() -> LibrarySettings:
    return _settings
