"""ADK context construction and event utilities.

Provides helpers for building ADK InvocationContext objects and
extracting citation URLs from ADK event streams. Used by both the
production API routes and the eval harness.
"""

from __future__ import annotations

from typing import Any

from core.logging import get_logger

log = get_logger(__name__)


def build_adk_context(query: str, query_id: str) -> tuple[Any, Any]:
    """Build a real ADK Session + InvocationContext for eval (no MagicMock).

    Returns (ctx, session) where ctx has a .session attribute.
    """
    from dataclasses import dataclass

    from google.adk.events import Event as ADKEvent
    from google.adk.sessions import Session
    from google.genai import types as genai_types

    user_event = ADKEvent(
        author="user",
        content=genai_types.Content(parts=[genai_types.Part(text=query)]),
    )
    session = Session(
        id=f"eval-{query_id}",
        app_name="eval",
        user_id="eval-runner",
        events=[user_event],
    )

    @dataclass
    class EvalInvocationContext:
        """Lightweight context for eval — no MagicMock."""

        session: Session

    return EvalInvocationContext(session=session), session


def extract_urls_from_adk_events(events: list[Any]) -> list[str]:
    """Extract citation URLs from ADK event custom_metadata."""
    urls: list[str] = []
    for event in events:
        meta = getattr(event, "custom_metadata", None) or {}
        # Citations from BedrockKB / Hybrid agents
        for citation in meta.get("citations", []):
            if isinstance(citation, dict) and citation.get("url"):
                urls.append(citation["url"])
        # Direct URLs from hybrid agent retrieved_urls
        for url in meta.get("retrieved_urls", []):
            if url:
                urls.append(url)
    return urls
