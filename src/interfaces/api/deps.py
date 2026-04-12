"""Dependency injection for the FastAPI application."""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from librarian.factory import create_librarian
from orchestration.nodes.generation import GeneratorAgent
from librarian.bedrock.client import BedrockKBClient
from librarian.ingestion.pipeline import IngestionPipeline
from librarian.config import LibrarySettings, settings as _default_settings
from interfaces.api.triage import TriageService
from core.llm import AnthropicLLM
from core.logging import get_logger

log = get_logger(__name__)

_graph: CompiledStateGraph | None = None
_generator_agent: GeneratorAgent | None = None
_pipeline: IngestionPipeline | None = None
_bedrock_client: BedrockKBClient | None = None
_triage: TriageService | None = None
_settings: LibrarySettings = _default_settings


def init_graph(cfg: LibrarySettings | None = None) -> None:
    """Initialise the graph singleton. Called once at app startup."""
    global _graph, _generator_agent, _bedrock_client, _settings  # noqa: PLW0603
    _settings = cfg or _default_settings

    log.info("api.deps.init_graph", retrieval=_settings.retrieval_strategy)
    _graph = create_librarian(_settings)

    llm = AnthropicLLM(
        model=_settings.anthropic_model_sonnet,
        api_key=_settings.anthropic_api_key,
    )
    _generator_agent = GeneratorAgent(
        llm=llm,
        confidence_threshold=_settings.confidence_threshold,
    )

    # Bedrock KB client — optional, only if configured
    if _settings.bedrock_knowledge_base_id:
        try:
            _bedrock_client = BedrockKBClient(_settings)
            log.info("api.deps.bedrock_kb.init", kb_id=_settings.bedrock_knowledge_base_id)
        except Exception:
            log.warning("api.deps.bedrock_kb.init_failed", exc_info=True)
            _bedrock_client = None
    else:
        log.info("api.deps.bedrock_kb.skipped", reason="no knowledge_base_id configured")


def get_graph() -> CompiledStateGraph:
    """Return the compiled LangGraph graph."""
    if _graph is None:
        msg = "Graph not initialised — call init_graph() first"
        raise RuntimeError(msg)
    return _graph


def get_generator_agent() -> GeneratorAgent:
    """Return the generator agent for streaming."""
    if _generator_agent is None:
        msg = "Generator agent not initialised — call init_graph() first"
        raise RuntimeError(msg)
    return _generator_agent


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


def is_graph_ready() -> bool:
    """Return True when the librarian graph singleton is initialised."""
    return _graph is not None


def is_bedrock_available() -> bool:
    """Return True when a Bedrock KB client is configured."""
    return _bedrock_client is not None


def init_triage() -> None:
    """Initialise the triage singleton. Called once at app startup."""
    global _triage  # noqa: PLW0603
    _triage = TriageService(
        graph_ready=is_graph_ready,
        bedrock_available=is_bedrock_available,
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
