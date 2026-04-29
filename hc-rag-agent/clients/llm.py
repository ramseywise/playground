"""LLM client factory — size-based resolution via LLM_PROVIDER.

Set LLM_PROVIDER to: gemini (default) | anthropic | openai

Sizes map to:
  small  → fast/cheap model (planner, hybrid probes, summarizer)
  medium → main chat model  (answer, reranker LLM listwise)
  large  → reasoning model  (reserved)

Usage:
    from clients.llm import get_chat_model, get_planner_chat_model, resolve_chat_model

    llm = get_chat_model()              # medium — main answer model
    planner = get_planner_chat_model()  # small  — planner / light probes
    llm = resolve_chat_model("large")   # explicit size
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Literal

import structlog

log = structlog.get_logger(__name__)

_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
_USE_LIGHT_PLANNER = os.getenv("RAG_PLANNER_LIGHT_MODEL", "false").lower() == "true"

_GEMINI = {
    "small": "gemini-2.0-flash-lite",
    "medium": "gemini-2.0-flash",
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
    return _REGISTRY.get(_PROVIDER, _GEMINI)[size]


@lru_cache(maxsize=12)
def resolve_chat_model(
    size: Literal["small", "medium", "large"] = "medium",
    temperature: float = 0,
) -> Any:
    model_id = resolve_model_id(size)
    log.debug("llm.resolve", provider=_PROVIDER, size=size, model=model_id)
    if _PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_id, temperature=temperature, streaming=True)
    if _PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic  # type: ignore[import]
        return ChatAnthropic(model=model_id, temperature=temperature, streaming=True)
    if _PROVIDER == "openai":
        from langchain_openai import ChatOpenAI  # type: ignore[import]
        return ChatOpenAI(model=model_id, temperature=temperature, streaming=True)
    raise ValueError(f"Unsupported LLM_PROVIDER: {_PROVIDER!r}. Use gemini | anthropic | openai")


def get_chat_model() -> Any:
    """Main answer/generation model (medium size)."""
    return resolve_chat_model("medium")


def get_planner_chat_model() -> Any:
    """Planner / light-probe model — small when RAG_PLANNER_LIGHT_MODEL=true, else medium."""
    return resolve_chat_model("small" if _USE_LIGHT_PLANNER else "medium")


def llm_configured() -> bool:
    if _PROVIDER == "gemini":
        return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    if _PROVIDER == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    if _PROVIDER == "openai":
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"))
    return False


def require_llm_for_cli() -> None:
    import sys
    if llm_configured():
        return
    log.error("llm.not_configured", provider=_PROVIDER)
    sys.exit(2)
