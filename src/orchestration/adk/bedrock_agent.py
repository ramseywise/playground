"""ADK agent wrapping AWS Bedrock Knowledge Bases.

Bedrock handles embedding, retrieval, and generation internally.
This agent extracts the user query from ADK session context,
forwards it to Bedrock, and emits the response as an ADK event.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from clients.bedrock import BedrockKBClient
from librarian.config import LibrarySettings
from core.logging import get_logger

log = get_logger(__name__)


def _extract_latest_query(ctx: InvocationContext) -> str:
    """Extract the latest user message text from ADK session events."""
    for event in reversed(ctx.session.events):
        if event.author == "user" and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    return part.text
    return ""


class BedrockKBAgent(BaseAgent):
    """ADK agent wrapping AWS Bedrock Knowledge Bases (managed RAG).

    Delegates entirely to BedrockKBClient — no custom retrieval logic.
    Useful for A/B comparison: ADK session management + Bedrock retrieval
    vs. the full Librarian LangGraph pipeline.
    """

    # Pydantic model fields — ADK BaseAgent is a Pydantic model
    _client: BedrockKBClient
    _cfg: LibrarySettings

    def __init__(self, cfg: LibrarySettings, **kwargs: Any) -> None:
        super().__init__(
            name="bedrock_kb",
            description="RAG via AWS Bedrock Knowledge Bases (managed)",
            **kwargs,
        )
        # Private attrs set after pydantic init
        object.__setattr__(self, "_client", BedrockKBClient(cfg))
        object.__setattr__(self, "_cfg", cfg)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Extract query from ADK session, forward to Bedrock, emit response."""
        query = _extract_latest_query(ctx)

        log.info(
            "adk.bedrock_kb.query",
            query=query[:80],
            session_id=ctx.session.id,
        )

        resp = await self._client.aquery(
            query,
            session_id=ctx.session.id,
        )

        log.info(
            "adk.bedrock_kb.response",
            response_len=len(resp.response),
            citation_count=len(resp.citations),
        )

        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=resp.response)],
            ),
        )
