"""Safe JSON extraction from LLM output (plain JSON or fenced markdown)."""

from __future__ import annotations

import json
from typing import Any


def _strip_code_fence(text: str) -> str:
    """If ``text`` opens with a markdown code fence, return inner body; else ``text``."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return ""
    body: list[str] = []
    for line in lines[1:]:
        if line.strip() == "```":
            break
        body.append(line)
    return "\n".join(body).strip()


def parse_json_safe(raw: str | None) -> Any:
    """Parse JSON from a string, optionally wrapped in ``` / ```json fences.

    Returns ``None`` on empty input or invalid JSON. Note: JSON ``null`` parses to
    Python ``None``, same as failure — callers cannot distinguish.
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    payload = _strip_code_fence(stripped)
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None
