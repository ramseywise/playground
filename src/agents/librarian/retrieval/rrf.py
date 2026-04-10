"""Reciprocal Rank Fusion (RRF) for multi-retriever result merging.

Implements the RRF algorithm (Cormack et al. 2009) to fuse multiple ranked
lists into a single ranking.  Useful when combining results from different
retrieval strategies (e.g. dense + BM25) before reranking.

Extracted from legacy ``reranker/reranker.py`` and adapted for the current
``GradedChunk`` schema.
"""

from __future__ import annotations

from hashlib import sha256

from agents.librarian.schemas.chunks import Chunk, GradedChunk

# Standard RRF smoothing constant (Cormack et al. 2009).
RRF_K: int = 60


def fuse_rankings(
    rankings: list[list[GradedChunk]],
    *,
    k: int = RRF_K,
    top_k: int | None = None,
) -> list[GradedChunk]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Each input list is assumed to be sorted by relevance (best first).
    Documents appearing in multiple lists accumulate RRF scores.

    Args:
        rankings: Ranked lists of ``GradedChunk`` to fuse.
        k: RRF smoothing parameter (default 60).
        top_k: If set, return only the top-k results.

    Returns:
        Fused list sorted by aggregated RRF score (descending).

    """
    if not rankings:
        return []

    chunk_scores: dict[str, float] = {}
    chunk_objects: dict[str, GradedChunk] = {}

    for ranking in rankings:
        for rank, graded in enumerate(ranking, 1):
            # Content-hash the chunk for dedup across lists.
            doc_hash = _chunk_hash(graded.chunk)
            rrf_score = 1.0 / (k + rank)
            chunk_scores[doc_hash] = chunk_scores.get(doc_hash, 0.0) + rrf_score
            # Keep the copy with the highest original score.
            if (
                doc_hash not in chunk_objects
                or graded.score > chunk_objects[doc_hash].score
            ):
                chunk_objects[doc_hash] = graded

    sorted_items = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

    results = [
        GradedChunk(
            chunk=chunk_objects[h].chunk,
            score=rrf_score,
            relevant=chunk_objects[h].relevant,
        )
        for h, rrf_score in sorted_items
    ]

    if top_k is not None:
        results = results[:top_k]
    return results


def _chunk_hash(chunk: Chunk) -> str:
    """Deterministic hash of a chunk for dedup across ranked lists."""
    return sha256(f"{chunk.id}:{chunk.text[:200]}".encode()).hexdigest()[:16]
