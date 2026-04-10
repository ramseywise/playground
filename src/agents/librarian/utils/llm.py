"""Thin anthropic SDK wrapper — replaces langchain-anthropic ChatAnthropic.

This keeps the librarian decoupled from the langchain ecosystem while
still using langgraph for orchestration (langgraph pulls langchain-core
transitively, but we don't import from it in production code paths).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import anthropic

from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)


class AnthropicLLM:
    """Direct anthropic SDK wrapper for LLM calls.

    Interface:
        ``generate(system, messages) -> str`` — full response
        ``stream(system, messages) -> AsyncIterator[str]`` — token-by-token

    Used by both the generation agent and the LLM listwise reranker.
    Replaces ``ChatAnthropic`` from ``langchain-anthropic`` — same
    functionality with zero LangChain dependency.
    """

    def __init__(self, model: str, api_key: str = "") -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or None,  # None → reads ANTHROPIC_API_KEY env var
        )
        self._model = model

    async def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
    ) -> str:
        """Call the anthropic messages API and return the response text."""
        response = await self._client.messages.create(
            model=self._model,
            system=system,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
        )
        text: str = response.content[0].text  # type: ignore[union-attr]
        log.debug(
            "llm.generate.done",
            model=self._model,
            input_messages=len(messages),
            output_chars=len(text),
        )
        return text

    async def stream(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Yield text chunks as they arrive from the Anthropic streaming API."""
        async with self._client.messages.stream(
            model=self._model,
            system=system,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
        ) as response_stream:
            async for text in response_stream.text_stream:
                yield text
        log.debug("llm.stream.done", model=self._model)

    @property
    def model(self) -> str:
        """Model ID used for API calls."""
        return self._model
