"""Query decomposition — split compound queries into sub-queries."""

from __future__ import annotations

import re

_CONJUNCTION_SPLIT = re.compile(
    r"\b(?:and|also|additionally|furthermore|moreover|plus)\b",
    re.I,
)
_QUESTION_SPLIT = re.compile(r"\?+")


def decompose_query(query: str) -> list[str]:
    """Split a compound query into sub-queries.

    Splits on question marks first, then on conjunctions.
    Always returns at least one element (the original query).
    """
    # Split on question marks first
    parts = [p.strip() for p in _QUESTION_SPLIT.split(query) if p.strip()]

    # Then split each part on conjunctions
    sub_queries: list[str] = []
    for part in parts:
        splits = [s.strip() for s in _CONJUNCTION_SPLIT.split(part) if s.strip()]
        sub_queries.extend(splits)

    if not sub_queries:
        return [query.strip()]

    return sub_queries
