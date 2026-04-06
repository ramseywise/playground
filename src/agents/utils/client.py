"""Thin Claude API client helper and shared LLM utilities."""

from __future__ import annotations

import json
from typing import Any

import anthropic
import structlog

from agents.utils.config import settings

log = structlog.get_logger(__name__)


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


def parse_json_response(
    client: anthropic.Anthropic,
    response_text: str,
    model: str,
    system: str,
) -> Any:
    """Parse JSON from a Claude response, retrying once on parse failure.

    On the first JSONDecodeError, re-prompts Claude with the malformed output
    and an instruction to return valid JSON. Raises on the second failure.
    """
    raw = strip_json_fences(response_text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("json.parse.retry", error=str(exc))
        retry_response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system,
            messages=[
                {"role": "user", "content": response_text},
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was not valid JSON: {exc}. "
                        "Please return only valid JSON with no other text."
                    ),
                },
            ],
        )
        raw = strip_json_fences(retry_response.content[0].text)
        return json.loads(raw)  # raise if still bad
