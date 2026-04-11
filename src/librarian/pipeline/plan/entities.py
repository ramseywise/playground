"""Entity extraction — regex-based pattern matching."""

from __future__ import annotations

import re

ENTITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("version", re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")),
    (
        "date",
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\b",
            re.I,
        ),
    ),
    (
        "quantity",
        re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|s|kb|mb|gb|tb|req|rpm|qps|%)\b", re.I),
    ),
    ("identifier", re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")),
]


def extract_entities(query: str) -> dict[str, list[str]]:
    """Extract typed entities from *query* using regex patterns."""
    result: dict[str, list[str]] = {}
    for label, pattern in ENTITY_PATTERNS:
        matches = pattern.findall(query)
        if matches:
            result[label] = list(dict.fromkeys(matches))  # deduplicate, preserve order
    return result
