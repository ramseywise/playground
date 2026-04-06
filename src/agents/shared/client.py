"""Thin Claude API client helper and shared LLM utilities."""

from __future__ import annotations

import anthropic

from agents.shared.config import settings


def create_client() -> anthropic.Anthropic:
    """Create an Anthropic client, validating the API key at call time."""
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — add it to .env")
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences from LLM JSON output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return text
