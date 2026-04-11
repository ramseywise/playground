"""Tests for GeminiLLM — mocked, no real API calls."""

from __future__ import annotations

import sys
import types as stdlib_types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub google.genai.types before importing GeminiLLM so the lazy imports
# inside generate() / generate_sync() resolve without the real SDK.
_genai_types = stdlib_types.ModuleType("google.genai.types")


class _FakeGenerateContentConfig:
    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


_genai_types.GenerateContentConfig = _FakeGenerateContentConfig  # type: ignore[attr-defined]

# Build the google → google.genai → google.genai.types module chain
_genai_mod = stdlib_types.ModuleType("google.genai")
_genai_mod.types = _genai_types  # type: ignore[attr-defined]
_google_mod = stdlib_types.ModuleType("google")
_google_mod.genai = _genai_mod  # type: ignore[attr-defined]

sys.modules.setdefault("google", _google_mod)
sys.modules.setdefault("google.genai", _genai_mod)
sys.modules.setdefault("google.genai.types", _genai_types)

from core.clients.llm import GeminiLLM, LLMClient, LLMClientSync, _to_gemini_contents


# ---------------------------------------------------------------------------
# Message format conversion
# ---------------------------------------------------------------------------


class TestToGeminiContents:
    def test_converts_assistant_to_model(self) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = _to_gemini_contents(messages)
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "model"

    def test_preserves_user_role(self) -> None:
        messages = [{"role": "user", "content": "test"}]
        result = _to_gemini_contents(messages)
        assert result[0]["role"] == "user"
        assert result[0]["parts"] == [{"text": "test"}]

    def test_empty_messages(self) -> None:
        assert _to_gemini_contents([]) == []


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestGeminiLLMProtocol:
    def test_implements_async_protocol(self) -> None:
        llm = GeminiLLM(model="gemini-2.0-flash", api_key="test")
        assert isinstance(llm, LLMClient)

    def test_implements_sync_protocol(self) -> None:
        llm = GeminiLLM(model="gemini-2.0-flash", api_key="test")
        assert isinstance(llm, LLMClientSync)

    def test_model_property(self) -> None:
        llm = GeminiLLM(model="gemini-2.0-flash", api_key="test")
        assert llm.model == "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Generate (async)
# ---------------------------------------------------------------------------


class TestGeminiGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_text(self) -> None:
        mock_response = MagicMock()
        mock_response.text = "Generated response"

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models

        llm = GeminiLLM(model="gemini-2.0-flash", api_key="test")
        llm._client = mock_client

        result = await llm.generate(
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result == "Generated response"
        mock_aio_models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_handles_none_text(self) -> None:
        mock_response = MagicMock()
        mock_response.text = None

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models

        llm = GeminiLLM(model="gemini-2.0-flash", api_key="test")
        llm._client = mock_client

        result = await llm.generate(
            system="system",
            messages=[{"role": "user", "content": "test"}],
        )
        assert result == ""


# ---------------------------------------------------------------------------
# Generate sync
# ---------------------------------------------------------------------------


class TestGeminiGenerateSync:
    def test_generate_sync_returns_text(self) -> None:
        mock_response = MagicMock()
        mock_response.text = "Sync response"

        mock_models = MagicMock()
        mock_models.generate_content = MagicMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.models = mock_models

        llm = GeminiLLM(model="gemini-2.0-flash", api_key="test")
        llm._client = mock_client

        result = llm.generate_sync(
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result == "Sync response"
        mock_models.generate_content.assert_called_once()
