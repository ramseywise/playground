"""LLM listwise reranker — uses the generation LLM to order chunks by relevance."""

from __future__ import annotations

import json
import logging

from rag.schemas.chunks import GradedChunk, RankedChunk

log = logging.getLogger(__name__)


def _parse(raw: str, chunks: list[GradedChunk]) -> list[RankedChunk] | None:
    """Parse a JSON list of 1-based ranks from the LLM response."""
    try:
        data = json.loads(raw.strip())
        if not isinstance(data, list):
            return None
        ranked: list[RankedChunk] = []
        for new_rank, old_rank in enumerate(data, start=1):
            idx = int(old_rank) - 1
            if 0 <= idx < len(chunks):
                ranked.append(
                    RankedChunk(
                        chunk=chunks[idx].chunk,
                        relevance_score=1.0 - (new_rank - 1) / max(len(data), 1),
                        rank=new_rank,
                    )
                )
        return ranked or None
    except Exception:
        return None


def _fallback(chunks: list[GradedChunk], top_k: int) -> list[RankedChunk]:
    top = sorted(chunks, key=lambda g: g.score, reverse=True)[:top_k]
    return [
        RankedChunk(chunk=g.chunk, relevance_score=g.score, rank=i + 1)
        for i, g in enumerate(top)
    ]


class LLMListwiseReranker:
    """Rerank using the generation LLM (:func:`src.clients.llm.get_chat_model`).

    The LLM is asked to return a JSON list of original 1-based positions
    sorted by relevance to the query, e.g. [3, 1, 2].
    """

    def __init__(self, top_k: int = 3) -> None:
        self._top_k = top_k

    async def rerank(
        self, query: str, chunks: list[GradedChunk], top_k: int | None = None
    ) -> list[RankedChunk]:
        import asyncio
        from clients.llm import get_chat_model

        k = top_k or self._top_k
        if not chunks:
            return []

        numbered = "\n".join(
            f"{i + 1}. {gc.chunk.text[:300]}" for i, gc in enumerate(chunks)
        )
        prompt = (
            f"Query: {query}\n\n"
            f"Passages:\n{numbered}\n\n"
            "Return a JSON array of the passage numbers ordered from most to least relevant. "
            "Example: [2, 1, 3]. Return JSON only."
        )

        try:
            response = await asyncio.to_thread(get_chat_model().invoke, prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            ranked = _parse(raw, chunks)
            if ranked:
                log.info("llm_listwise.rerank candidates=%d top_k=%d", len(chunks), k)
                return ranked[:k]
        except Exception as e:
            log.warning("llm_listwise.rerank.failed error=%s fallback=score_order", e)

        return _fallback(chunks, k)
