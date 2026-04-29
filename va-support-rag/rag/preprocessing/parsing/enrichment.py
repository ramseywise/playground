"""Metadata enrichment for document corpora."""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)


def extract_metadata(text: str, url: str | None = None) -> dict[str, Any]:
    """Extract lightweight metadata from document text and URL.

    Returns a dict with: word_count, char_count, and optionally source / type
    fields inferred from URL structure and content heuristics.

    Args:
        text: Document text.
        url: Optional source URL used to infer source category.

    Returns:
        Metadata dict (never raises).
    """
    meta: dict[str, Any] = {
        "word_count": len(text.split()),
        "char_count": len(text),
    }

    if url:
        if "/blog" in url:
            meta["source"] = "blog"
        elif "/api" in url or "/reference" in url:
            meta["source"] = "api_docs"
        elif "/help" in url or "/support" in url or "/docs" in url:
            meta["source"] = "help_center"

        id_match = re.search(r"/articles?/(\d+)", url)
        if id_match:
            meta["article_id"] = id_match.group(1)

    # Content-type heuristics
    if re.search(r"\bstep\s+\d+\b|\bhow\s+to\b", text, re.IGNORECASE):
        meta["content_type"] = "procedural"
    elif re.search(r"\bfaq\b|^\s*q\s*:", text, re.IGNORECASE | re.MULTILINE):
        meta["content_type"] = "faq"

    return meta


def enrich_documents(documents: list[dict], text_field: str = "text") -> list[dict]:
    """Enrich documents with metadata extracted from text and URL.

    Merges extract_metadata() results into each doc dict (does not overwrite
    existing keys).

    Args:
        documents: List of document dicts.
        text_field: Field containing text.

    Returns:
        New list of dicts with metadata fields added.
    """
    enriched: list[dict] = []
    for doc in documents:
        text = doc.get(text_field) or doc.get("content", "")
        url = doc.get("url", "")
        meta = extract_metadata(text, url)
        enriched.append({**meta, **doc})  # doc wins on key conflicts

    log.info("parsing.enrich.done n=%d", len(enriched))
    return enriched
