"""Shared utilities for ADK agents."""

from __future__ import annotations

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event


def extract_latest_query(ctx: InvocationContext) -> str:
    """Extract the latest user message text from ADK session events."""
    for event in reversed(ctx.session.events):
        if event.author == "user" and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    return part.text
    return ""


def extract_messages(events: list[Event]) -> list[dict[str, str]]:
    """Convert ADK session events into LangGraph-compatible message dicts."""
    messages: list[dict[str, str]] = []
    for event in events:
        if not event.content or not event.content.parts:
            continue
        text = ""
        for part in event.content.parts:
            if hasattr(part, "text") and part.text:
                text += part.text
        if not text:
            continue
        role = "user" if event.author == "user" else "assistant"
        messages.append({"role": role, "content": text})
    return messages
