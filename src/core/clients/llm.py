"""Provider-agnostic LLM client.

Defines ``LLMClient`` (async) and ``LLMClientSync`` (sync) protocols,
plus ``AnthropicLLM`` and ``GeminiLLM`` implementations.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

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

    async def stream(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Yield text chunks as they arrive from the Anthropic streaming API."""
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic(api_key=self._api_key)
        async with self._async_client.messages.stream(
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


# ---------------------------------------------------------------------------
# Gemini implementation
# ---------------------------------------------------------------------------


def _to_gemini_contents(
    messages: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Convert Anthropic-style messages to Gemini content format.

    Gemini uses ``"model"`` where Anthropic uses ``"assistant"``.
    """
    return [
        {
            "role": "model" if m["role"] == "assistant" else m["role"],
            "parts": [{"text": m["content"]}],
        }
        for m in messages
    ]


class GeminiLLM:
    """Google Gemini SDK wrapper implementing both sync and async LLM protocols.

    When *api_key* is empty the SDK falls back to the ``GOOGLE_API_KEY`` env var.
    Requires the ``google-genai`` package (install via ``uv sync --extra gemini``).
    """

    def __init__(self, model: str, api_key: str = "") -> None:
        self._api_key = api_key or None  # None → SDK reads GOOGLE_API_KEY env var
        self._model = model
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai  # type: ignore[import-untyped]

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    # -- async (LLMClient) -------------------------------------------------

    async def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
    ) -> str:
        """Call the Gemini API (async) and return response text."""
        from google.genai import types  # type: ignore[import-untyped]

        client = self._get_client()
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )
        response = await client.aio.models.generate_content(
            model=self._model,
            contents=_to_gemini_contents(messages),
            config=config,
        )
        try:
            text = response.text or ""
        except ValueError:
            # Gemini raises ValueError when response is safety-blocked
            log.warning(
                "llm.gemini.generate.safety_blocked",
                model=self._model,
                finish_reason=getattr(
                    getattr(response, "candidates", [None])[0] if response.candidates else None,
                    "finish_reason",
                    "unknown",
                ),
            )
            return ""
        log.debug(
            "llm.gemini.generate.done",
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
        """Call the Gemini API (sync) and return response text."""
        from google.genai import types  # type: ignore[import-untyped]

        client = self._get_client()
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )
        response = client.models.generate_content(
            model=self._model,
            contents=_to_gemini_contents(messages),
            config=config,
        )
        try:
            text = response.text or ""
        except ValueError:
            log.warning(
                "llm.gemini.generate_sync.safety_blocked",
                model=self._model,
                finish_reason=getattr(
                    getattr(response, "candidates", [None])[0] if response.candidates else None,
                    "finish_reason",
                    "unknown",
                ),
            )
            return ""
        log.debug(
            "llm.gemini.generate_sync.done",
            model=self._model,
            input_messages=len(messages),
            output_chars=len(text),
        )
        return text

    @property
    def model(self) -> str:
        """Model ID used for API calls."""
        return self._model
