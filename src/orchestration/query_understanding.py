"""Backward-compatible re-export — canonical location is ``orchestration.langgraph.query_understanding``."""

from __future__ import annotations

from orchestration.langgraph.query_understanding import (  # noqa: F401
    QueryAnalysis,
    QueryAnalyzer,
    QueryRouter,
    TERM_EXPANSIONS,
)

__all__ = ["QueryAnalysis", "QueryAnalyzer", "QueryRouter", "TERM_EXPANSIONS"]
