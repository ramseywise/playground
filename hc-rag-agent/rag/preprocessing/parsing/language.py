"""Heuristic language detection and filtering."""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)


def detect_language(text: str) -> dict[str, Any]:
    """Heuristic language detection — does not require an external library.

    Returns a dict with keys:
        language: "unknown" | "uncertain" | detected language code(s)
        confidence: float [0, 1]
        issues: list of detected non-target script names

    Note: This is a lightweight heuristic for filtering, not a replacement for
    langdetect/fasttext. For production multi-language corpora use a proper detector.
    """
    if not text or len(text.strip()) < 10:
        return {"language": "unknown", "confidence": 0.0, "issues": ["too_short"]}

    issues: list[str] = []

    if re.search(r"[\u0400-\u04FF]", text):
        issues.append("cyrillic")
    if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text):
        issues.append("cjk")
    if re.search(r"[\u0600-\u06FF]", text):
        issues.append("arabic")

    if issues:
        return {"language": "non_latin", "confidence": 0.9, "issues": issues}

    return {"language": "latin", "confidence": 0.8, "issues": []}


def filter_by_language(
    documents: list[dict],
    allowed_languages: set[str] | None = None,
    text_field: str = "text",
) -> list[dict]:
    """Filter documents by detected language.

    Args:
        documents: List of document dicts.
        allowed_languages: Set of allowed language codes (e.g. {"latin", "uncertain"}).
            Defaults to {"latin", "unknown"} — keeps anything without a detected script issue.
        text_field: Field name containing text content.

    Returns:
        Filtered list of documents.
    """
    if allowed_languages is None:
        allowed_languages = {"latin", "unknown"}

    kept: list[dict] = []
    filtered = 0

    for doc in documents:
        text = doc.get(text_field) or doc.get("content", "")
        lang = detect_language(text)
        if lang["language"] in allowed_languages:
            kept.append(doc)
        else:
            filtered += 1

    log.info(
        "parsing.filter_language.done kept=%d filtered=%d",
        len(kept),
        filtered,
    )
    return kept
