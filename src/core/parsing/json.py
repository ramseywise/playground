"""JSON parsing helpers for LLM output.

Handles markdown code fences and retry-on-parse-failure logic that is
shared across agents (presenter, librarian, etc.).
"""

from __future__ import annotations

import json
from typing import Any

import structlog

log = structlog.get_logger(__name__)


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences from LLM JSON output."""
    text = text.strip()
    if text.startswith("```"):
        text = (
            text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        )
    return text


def parse_json_response(raw_text: str) -> Any:
    """Parse JSON from an LLM response, stripping code fences first.

    Raises ``json.JSONDecodeError`` if the text is not valid JSON.
    """
    return json.loads(strip_json_fences(raw_text))
