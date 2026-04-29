"""Reciprocal Rank Fusion — merges multiple ranked lists into a single ranking."""

from __future__ import annotations

from hashlib import sha256

from rag.schemas.chunks import GradedChunk


def chunk_fingerprint(gc: GradedChunk) -> str:
    """Content fingerprint: ``url|text[:200]`` → SHA-256 prefix. Used for dedup."""
    raw = f"{gc.chunk.metadata.url}|{gc.chunk.text[:200].lower().strip()}"
    return sha256(raw.encode()).hexdigest()[:16]


def fuse_rankings(lists: list[list[GradedChunk]], k: int = 60) -> list[GradedChunk]:
    """Reciprocal Rank Fusion across multiple ranked lists.

    Score formula: Σ 1 / (k + rank_i) for each list where the chunk appears.
    Deduplicates by content fingerprint (keeping highest-scored copy).
    Returns chunks sorted by fused score descending.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, GradedChunk] = {}

    for ranked_list in lists:
        for rank, gc in enumerate(ranked_list, start=1):
            h = chunk_fingerprint(gc)
            scores[h] = scores.get(h, 0.0) + 1.0 / (k + rank)
            if h not in chunks or gc.score > chunks[h].score:
                chunks[h] = gc

    return sorted(
        chunks.values(), key=lambda gc: scores[chunk_fingerprint(gc)], reverse=True
    )
