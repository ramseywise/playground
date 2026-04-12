"""Backward-compatible re-export — canonical location is ``orchestration.langgraph.graph``."""

from __future__ import annotations

from orchestration.langgraph.graph import (  # noqa: F401
    build_graph,
    _route_after_analyze,
    _route_after_gate,
)

__all__ = ["build_graph", "_route_after_analyze", "_route_after_gate"]
