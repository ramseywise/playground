"""Model factory — resolves LLM_PROVIDER + size to a model ID string.

Usage:
    from shared.model_factory import resolve_model_id

    model_id = resolve_model_id("medium")   # "gemini-2.5-flash"

Set LLM_PROVIDER env var to switch providers: gemini (default) | anthropic | openai
ADK agents consume model IDs as strings — instantiation is handled by the ADK runner.
"""

from __future__ import annotations

import os
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
