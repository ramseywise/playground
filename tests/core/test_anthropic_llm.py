"""Tests for AnthropicLLM — mocked, no real API calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.clients.llm import AnthropicLLM, LLMClient, LLMClientSync


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestAnthropicLLMProtocol:
    def test_implements_async_protocol(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        assert isinstance(llm, LLMClient)

    def test_implements_sync_protocol(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        assert isinstance(llm, LLMClientSync)

    def test_model_property(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        assert llm.model == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# max_retries wired to SDK constructors
# ---------------------------------------------------------------------------


class TestAnthropicRetry:
    @pytest.mark.asyncio
    async def test_async_client_created_with_max_retries(self) -> None:
        """AsyncAnthropic is constructed with max_retries=3."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]

        with patch("core.clients.llm.anthropic.AsyncAnthropic") as mock_cls:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client

            llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test-key")
            result = await llm.generate(
                system="system",
                messages=[{"role": "user", "content": "hi"}],
            )

            mock_cls.assert_called_once_with(api_key="test-key", max_retries=3)
            assert result == "ok"

    def test_sync_client_created_with_max_retries(self) -> None:
        """Anthropic (sync) is constructed with max_retries=3."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]

        with patch("core.clients.llm.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create = MagicMock(return_value=mock_response)
            mock_cls.return_value = mock_client

            llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test-key")
            result = llm.generate_sync(
                system="system",
                messages=[{"role": "user", "content": "hi"}],
            )

            mock_cls.assert_called_once_with(api_key="test-key", max_retries=3)
            assert result == "ok"

    @pytest.mark.asyncio
    async def test_stream_client_created_with_max_retries(self) -> None:
        """AsyncAnthropic for streaming is also created with max_retries=3."""
        with patch("core.clients.llm.anthropic.AsyncAnthropic") as mock_cls:
            # Build a mock async context manager for messages.stream
            mock_stream_cm = AsyncMock()
            mock_stream_cm.__aenter__ = AsyncMock()
            mock_text_stream = AsyncMock()
            mock_text_stream.__aiter__ = lambda self: self
            mock_text_stream.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
            mock_stream_cm.__aenter__.return_value = MagicMock(
                text_stream=mock_text_stream
            )
            mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

            mock_client = MagicMock()
            mock_client.messages.stream = MagicMock(return_value=mock_stream_cm)
            mock_cls.return_value = mock_client

            llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test-key")
            chunks = []
            async for chunk in llm.stream(
                system="system",
                messages=[{"role": "user", "content": "hi"}],
            ):
                chunks.append(chunk)

            mock_cls.assert_called_once_with(api_key="test-key", max_retries=3)


# ---------------------------------------------------------------------------
# Generate (async)
# ---------------------------------------------------------------------------


class TestAnthropicGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_text(self) -> None:
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated response")]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        llm._async_client = mock_client

        result = await llm.generate(
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result == "Generated response"
        mock_client.messages.create.assert_called_once()


# ---------------------------------------------------------------------------
# Generate sync
# ---------------------------------------------------------------------------


class TestAnthropicGenerateSync:
    def test_generate_sync_returns_text(self) -> None:
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Sync response")]

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_response)

        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        llm._sync_client = mock_client

        result = llm.generate_sync(
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result == "Sync response"
        mock_client.messages.create.assert_called_once()
