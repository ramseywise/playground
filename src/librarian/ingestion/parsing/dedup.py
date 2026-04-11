"""Exact and fuzzy deduplication for document corpora."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from core.logging import get_logger

log = get_logger(__name__)


def compute_text_hash(text: str, normalize: bool = True) -> str:
    """MD5 hash of text for exact-duplicate detection.

    Args:
        text: Input text.
        normalize: If True, lowercases and collapses whitespace before hashing.

    Returns:
        Hex MD5 digest.
    """
    if normalize:
        text = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(text.encode()).hexdigest()


def deduplicate_exact(
    documents: list[dict],
    text_field: str = "text",
    keep: str = "first",
) -> list[dict]:
    """Remove exact-duplicate documents by text hash.

    Args:
        documents: List of document dicts.
        text_field: Field containing text to hash.
        keep: "first" keeps earliest occurrence; "last" keeps latest.

    Returns:
        Deduplicated list preserving original order.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    duplicates = 0

    items = documents if keep == "first" else list(reversed(documents))

    for doc in items:
        text = doc.get(text_field) or doc.get("content", "")
        h = compute_text_hash(text)
        if h not in seen:
            seen.add(h)
            unique.append(doc)
        else:
            duplicates += 1

    result = unique if keep == "first" else list(reversed(unique))
    log.info("parsing.dedup_exact.done", kept=len(result), removed=duplicates)
    return result


def deduplicate_fuzzy(
    documents: list[dict],
    embedder: Any,
    text_field: str = "text",
    threshold: float = 0.95,
) -> list[dict]:
    """Remove near-duplicate documents using embedding cosine similarity.

    Requires scikit-learn. Embeds all documents and removes any pair with
    cosine similarity >= threshold, keeping the earlier document.

    Args:
        documents: List of document dicts.
        embedder: Object with an embed_passages(texts) method returning list[list[float]].
        text_field: Field containing text to embed.
        threshold: Cosine similarity threshold for near-duplicates (default 0.95).

    Returns:
        Deduplicated list.
    """
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

    texts = [doc.get(text_field) or doc.get("content", "") for doc in documents]
    embeddings = embedder.embed_passages([t[:500] for t in texts])

    sim_matrix = cosine_similarity(embeddings)
    to_remove: set[int] = set()

    for i in range(len(sim_matrix)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(sim_matrix)):
            if sim_matrix[i, j] >= threshold:
                to_remove.add(j)

    result = [doc for i, doc in enumerate(documents) if i not in to_remove]
    log.info("parsing.dedup_fuzzy.done", kept=len(result), removed=len(to_remove))
    return result
