"""Model factory — resolves LLM_PROVIDER + size to a LangChain chat model or model ID.

Usage:
    from shared.model_factory import resolve_model_id, resolve_chat_model

    model_id = resolve_model_id("medium")          # "gemini-2.5-flash"
    llm = resolve_chat_model("small")              # cached ChatGoogleGenerativeAI
    llm = resolve_chat_model("medium", temperature=0.2)

Set LLM_PROVIDER env var to switch providers: gemini (default) | anthropic | openai
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

_GEMINI = {
    "small": "gemini-2.0-flash-lite",
    "medium": "gemini-2.5-flash",
    "large": "gemini-2.5-pro",
}
_ANTHROPIC = {
    "small": "claude-haiku-4-5-20251001",
    "medium": "claude-sonnet-4-6",
    "large": "claude-opus-4-7",
}
_OPENAI = {
    "small": "gpt-4o-mini",
    "medium": "gpt-4o",
    "large": "o1",
}
_REGISTRY: dict[str, dict[str, str]] = {
    "gemini": _GEMINI,
    "anthropic": _ANTHROPIC,
    "openai": _OPENAI,
}


def resolve_model_id(size: Literal["small", "medium", "large"] = "medium") -> str:
    """Return the model string for the configured LLM_PROVIDER and size tier."""
    return _REGISTRY.get(_PROVIDER, _GEMINI)[size]


@lru_cache(maxsize=12)
def resolve_chat_model(
    size: Literal["small", "medium", "large"] = "medium",
    temperature: float = 0,
):
    """Return a cached LangChain chat model for the configured LLM_PROVIDER.

    Results are cached by (size, temperature) so repeated calls reuse the same
    object — important for Gemini's server-side prefix caching of system prompts.
    """
    model_id = resolve_model_id(size)
    if _PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_id, temperature=temperature)
    if _PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic  # type: ignore[import]

        return ChatAnthropic(model=model_id, temperature=temperature)
    if _PROVIDER == "openai":
        from langchain_openai import ChatOpenAI  # type: ignore[import]

        return ChatOpenAI(model=model_id, temperature=temperature)
    raise ValueError(f"Unsupported LLM_PROVIDER: {_PROVIDER!r}")
