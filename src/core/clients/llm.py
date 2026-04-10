"""Provider-agnostic LLM client.

Defines ``LLMClient`` (async) and ``LLMClientSync`` (sync) protocols,
plus the ``AnthropicLLM`` implementation that satisfies both.

Future providers (Vercel AI, Google ADK) implement the same protocols.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import anthropic
import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Async LLM interface — used by the librarian and async agents."""

    async def generate(
        self, system: str, messages: list[dict[str, str]], max_tokens: int = 4096
    ) -> str: ...

    @property
    def model(self) -> str: ...


@runtime_checkable
class LLMClientSync(Protocol):
    """Sync LLM interface — used by researcher, presenter, cartographer."""

    def generate_sync(
        self, system: str, messages: list[dict[str, str]], max_tokens: int = 4096
    ) -> str: ...

    @property
    def model(self) -> str: ...


# ---------------------------------------------------------------------------
# Anthropic implementation
# ---------------------------------------------------------------------------


class AnthropicLLM:
    """Anthropic SDK wrapper implementing both sync and async LLM protocols.

    Reads ``ANTHROPIC_API_KEY`` from the environment when *api_key* is empty.
    """

    def __init__(self, model: str, api_key: str = "") -> None:
        self._api_key = api_key or None  # None → SDK reads env var
        self._model = model
        self._async_client: anthropic.AsyncAnthropic | None = None
        self._sync_client: anthropic.Anthropic | None = None

    # -- async (LLMClient) -------------------------------------------------

    async def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
    ) -> str:
        """Call the Anthropic messages API (async) and return response text."""
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic(api_key=self._api_key)
        response = await self._async_client.messages.create(
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

    # -- sync (LLMClientSync) ----------------------------------------------

    def generate_sync(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
    ) -> str:
        """Call the Anthropic messages API (sync) and return response text."""
        if self._sync_client is None:
            self._sync_client = anthropic.Anthropic(api_key=self._api_key)
        response = self._sync_client.messages.create(
            model=self._model,
            system=system,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
        )
        text: str = response.content[0].text  # type: ignore[union-attr]
        log.debug(
            "llm.generate_sync.done",
            model=self._model,
            input_messages=len(messages),
            output_chars=len(text),
        )
        return text

    @property
    def model(self) -> str:
        """Model ID used for API calls."""
        return self._model
