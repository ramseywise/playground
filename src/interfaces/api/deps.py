"""Dependency injection for the FastAPI application."""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from librarian.factory import create_librarian
from orchestration.nodes.generation import GenerationSubgraph
from librarian.bedrock.client import BedrockKBClient
from librarian.ingestion.pipeline import IngestionPipeline
from librarian.config import LibrarySettings, settings as _default_settings
from core.llm import AnthropicLLM
from core.logging import get_logger

log = get_logger(__name__)

_graph: CompiledStateGraph | None = None
_generation_sg: GenerationSubgraph | None = None
_pipeline: IngestionPipeline | None = None
_bedrock_client: BedrockKBClient | None = None
_settings: LibrarySettings = _default_settings


def init_graph(cfg: LibrarySettings | None = None) -> None:
    """Initialise the graph singleton. Called once at app startup."""
    global _graph, _generation_sg, _bedrock_client, _settings  # noqa: PLW0603
    _settings = cfg or _default_settings

    log.info("api.deps.init_graph", retrieval=_settings.retrieval_strategy)
    _graph = create_librarian(_settings)

    llm = AnthropicLLM(
        model=_settings.anthropic_model_sonnet,
        api_key=_settings.anthropic_api_key,
    )
    _generation_sg = GenerationSubgraph(
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


def get_generation_subgraph() -> GenerationSubgraph:
    """Return the generation subgraph for streaming."""
    if _generation_sg is None:
        msg = "Generation subgraph not initialised — call init_graph() first"
        raise RuntimeError(msg)
    return _generation_sg


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


def get_settings() -> LibrarySettings:
    return _settings
