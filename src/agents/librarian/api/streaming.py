"""SSE event formatting for the streaming chat endpoint."""

from __future__ import annotations

import json
from typing import Any


def format_sse(event: str, data: Any) -> str:
    """Format a single SSE event string.

    Returns a string like::

        event: token
        data: {"text": "Hello"}

    """
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"
