"""Unit tests for LLM client wiring (mocked providers — no API calls)."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from clients import llm


@pytest.fixture(autouse=True)
def clear_llm_cache() -> Generator[None, None, None]:
    llm.get_chat_model.cache_clear()
    yield
    llm.get_chat_model.cache_clear()


def test_llm_configured_gemini_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "GOOGLE_API_KEY", "x")
    assert llm.llm_configured() is True


def test_llm_configured_gemini_false_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "GOOGLE_API_KEY", None)
    assert llm.llm_configured() is False


def test_get_chat_model_anthropic_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(llm, "ANTHROPIC_API_KEY", "test-key")
    with patch(
        "langchain_anthropic.ChatAnthropic", return_value=MagicMock(name="claude")
    ) as ctor:
        model = llm.get_chat_model()
    ctor.assert_called_once()
    assert model is not None


def test_get_chat_model_openai_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm, "API_KEY", "sk-test")
    with patch(
        "langchain_openai.ChatOpenAI", return_value=MagicMock(name="gpt")
    ) as ctor:
        model = llm.get_chat_model()
    ctor.assert_called_once()
    assert model is not None


def test_get_chat_model_gemini_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "GOOGLE_API_KEY", "gk-test")
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
    monkeypatch.setattr(llm, "LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        llm.get_chat_model()
