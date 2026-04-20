"""Canonical backend identifiers — single source of truth.

Every module that needs backend IDs, labels, or type literals should import
from here to avoid divergence.
"""

from __future__ import annotations

from typing import Literal

#: All backend IDs that the API supports.
BACKEND_IDS: tuple[str, ...] = (
    "librarian",
    "bedrock",
    "google_adk",
    "adk_bedrock",
    "adk_custom_rag",
    "adk_hybrid",
)

#: Literal union for Pydantic models and triage routing.
BackendLiteral = Literal[
    "librarian",
    "bedrock",
    "google_adk",
    "adk_bedrock",
    "adk_custom_rag",
    "adk_hybrid",
]

#: Triage routes include backend IDs plus internal-only routes.
Route = Literal[
    "librarian",
    "bedrock",
    "google_adk",
    "adk_bedrock",
    "adk_custom_rag",
    "adk_hybrid",
    "escalation",
    "direct",
]

#: Human-readable labels for the /backends discovery endpoint.
BACKEND_LABELS: dict[str, str] = {
    "librarian": "Custom RAG (LangGraph CRAG)",
    "bedrock": "AWS Bedrock KB",
    "google_adk": "Google Gemini + Vertex AI Search",
    "adk_bedrock": "ADK + Bedrock KB",
    "adk_custom_rag": "ADK + Custom RAG (Gemini)",
    "adk_hybrid": "ADK + LangGraph Hybrid",
}
