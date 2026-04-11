"""Intent classification — keyword-based rule matching."""

from __future__ import annotations

import re

from agents.librarian.rag_core.schemas.retrieval import Intent

# Order matters — first match wins.
INTENT_RULES: list[tuple[list[str], Intent]] = [
    (
        [
            "compare",
            "vs",
            "versus",
            "difference",
            "differences",
            "differ",
            "contrast",
            "which is better",
            "tradeoff",
            "pros and cons",
        ],
        Intent.COMPARE,
    ),
    (
        [
            "hello",
            "hi",
            "hey",
            "thanks",
            "thank you",
            "help me",
            "what can you do",
            "who are you",
            "good morning",
            "good afternoon",
        ],
        Intent.CONVERSATIONAL,
    ),
    (
        ["weather", "stock price", "sports score", "recipe", "movie"],
        Intent.OUT_OF_SCOPE,
    ),
    (
        [
            "overview",
            "explore",
            "survey",
            "summarize",
            "summary",
            "explain",
            "how does",
            "what is",
            "walk me through",
            "tell me about",
        ],
        Intent.EXPLORE,
    ),
]


def classify_intent(query_lower: str) -> tuple[Intent, float]:
    """Return (intent, confidence) for the given lowercased query."""
    for keywords, intent in INTENT_RULES:
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, query_lower):
                return intent, 0.9

    # Default: lookup with moderate confidence
    return Intent.LOOKUP, 0.6
