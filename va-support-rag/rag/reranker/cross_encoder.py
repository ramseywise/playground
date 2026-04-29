"""Cross-encoder reranker backed by sentence-transformers."""

from __future__ import annotations

import asyncio
import logging
import math
from functools import lru_cache
from typing import Any

from rag.schemas.chunks import GradedChunk, RankedChunk

log = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str) -> Any:
    from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]  # lazy import — heavy dep

    return CrossEncoder(model_name)


class CrossEncoderReranker:
    """ms-marco-MiniLM cross-encoder reranker."""

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        self._model_name = model_name

    async def rerank(
        self, query: str, chunks: list[GradedChunk], top_k: int = 3
    ) -> list[RankedChunk]:
        if not chunks:
            return []

        model = await asyncio.to_thread(_load_cross_encoder, self._model_name)
        pairs = [[query, gc.chunk.text] for gc in chunks]
        raw_scores = await asyncio.to_thread(model.predict, pairs)

        scored = sorted(
            zip(chunks, raw_scores),
            key=lambda t: float(t[1]),
            reverse=True,
        )

        results = [
            RankedChunk(
                chunk=gc.chunk, relevance_score=_sigmoid(float(score)), rank=i + 1
            )
            for i, (gc, score) in enumerate(scored[:top_k])
        ]

        log.info("cross_encoder.rerank candidates=%d top_k=%d", len(chunks), top_k)
        return results
