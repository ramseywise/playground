"""Managed RAG API clients — thin wrappers around external services.

These are framework-agnostic API wrappers, not orchestration logic.
Each client takes a ``LibrarySettings`` and exposes ``query()`` / ``aquery()``.
"""

from __future__ import annotations

from clients.bedrock_KB import BedrockKBClient, BedrockKBResponse
from clients.google_vertex import GoogleRAGClient, GoogleRAGResponse

__all__ = [
    "BedrockKBClient",
    "BedrockKBResponse",
    "GoogleRAGClient",
    "GoogleRAGResponse",
]
