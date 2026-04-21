"""AgentState — the single state object threaded through every graph node."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ── conversation ──────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    user_id: str                          # identifies user across sessions
    page_url: str | None

    # ── memory ────────────────────────────────────────────────────
    user_preferences: list[dict[str, str]]  # [{key, value}] loaded at turn start

    # ── routing ───────────────────────────────────────────────────
    intent: str | None
    routing_confidence: float

    # ── domain work ───────────────────────────────────────────────
    tool_results: list[dict[str, Any]]   # accumulated tool call results

    # ── final output ─────────────────────────────────────────────
    response: dict | None                # serialised AssistantResponse

    # ── safety ────────────────────────────────────────────────────
    blocked: bool
    block_reason: str | None
