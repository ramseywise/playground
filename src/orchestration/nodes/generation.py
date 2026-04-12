"""Backward-compatible re-export — canonical location is ``orchestration.langgraph.nodes.generation``."""

from __future__ import annotations

from orchestration.langgraph.nodes.generation import (  # noqa: F401
    DEFAULT_CONFIDENCE_GATE,
    GeneratorAgent,
    GenerationSubgraph,
)

__all__ = ["DEFAULT_CONFIDENCE_GATE", "GeneratorAgent", "GenerationSubgraph"]
