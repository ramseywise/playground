"""Unit tests for LLM client wiring (mocked providers — no API calls)."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from clients import llm


@pytest.fixture(autouse=True)
def clear_llm_cache() -> Generator[None, None, None]:
    llm.resolve_chat_model.cache_clear()
    yield
    llm.resolve_chat_model.cache_clear()


def test_llm_configured_gemini_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert llm.llm_configured() is True


def test_llm_configured_gemini_false_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "_PROVIDER", "gemini")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert llm.llm_configured() is False


def test_get_chat_model_anthropic_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_PROVIDER", "anthropic")
    fake_module = MagicMock()
    fake_ctor = MagicMock(return_value=MagicMock(name="claude"))
    fake_module.ChatAnthropic = fake_ctor
    with patch.dict("sys.modules", {"langchain_anthropic": fake_module}):
        model = llm.get_chat_model()
    fake_ctor.assert_called_once()
    assert model is not None


def test_get_chat_model_openai_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_PROVIDER", "openai")
    fake_module = MagicMock()
    fake_ctor = MagicMock(return_value=MagicMock(name="gpt"))
    fake_module.ChatOpenAI = fake_ctor
    with patch.dict("sys.modules", {"langchain_openai": fake_module}):
        model = llm.get_chat_model()
    fake_ctor.assert_called_once()
    assert model is not None


def test_get_chat_model_gemini_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_PROVIDER", "gemini")
    with patch(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        return_value=MagicMock(name="gemini"),
    ) as ctor:
        model = llm.get_chat_model()
    ctor.assert_called_once()
    assert model is not None


def test_get_chat_model_unknown_provider_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        llm.get_chat_model()
