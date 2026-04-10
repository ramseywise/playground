"""Dependency injection for the FastAPI application."""

from __future__ import annotations

from typing import Any

from agents.librarian.factory import create_librarian
from agents.librarian.orchestration.subgraphs.generation import GenerationSubgraph
from agents.librarian.utils.config import LibrarySettings, settings as _default_settings
from agents.librarian.utils.llm import AnthropicLLM
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)

_graph: Any = None
_generation_sg: GenerationSubgraph | None = None
_pipeline: Any = None
_settings: LibrarySettings = _default_settings


def init_graph(cfg: LibrarySettings | None = None) -> None:
    """Initialise the graph singleton. Called once at app startup."""
    global _graph, _generation_sg, _settings  # noqa: PLW0603
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


def get_graph() -> Any:
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
    from agents.librarian.factory import create_ingestion_pipeline

    _pipeline = create_ingestion_pipeline(cfg or _settings)
    log.info("api.deps.init_pipeline")


def get_pipeline() -> Any:
    """Return the ingestion pipeline."""
    if _pipeline is None:
        msg = "Pipeline not initialised — call init_pipeline() first"
        raise RuntimeError(msg)
    return _pipeline


def get_settings() -> LibrarySettings:
    return _settings
