from __future__ import annotations

import math
from typing import Any

from agents.librarian.rag_core.schemas.chunks import GradedChunk, RankedChunk
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_MODEL_CACHE: dict[str, Any] = {}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _load_cross_encoder(model_name: str) -> Any:
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

        log.info("reranker.cross_encoder.load", model=model_name)
        _MODEL_CACHE[model_name] = CrossEncoder(model_name)
    return _MODEL_CACHE[model_name]


class CrossEncoderReranker:
    """Reranker using a cross-encoder model (default: ms-marco-MiniLM-L-6-v2).

    Scores each (query, chunk.text) pair, applies sigmoid to logit → [0, 1].
    Model is loaded once at instantiation and cached process-wide.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = _load_cross_encoder(model_name)

    async def rerank(
        self,
        query: str,
        chunks: list[GradedChunk],
        top_k: int = 3,
    ) -> list[RankedChunk]:
        if not chunks:
            return []

        pairs = [[query, gc.chunk.text] for gc in chunks]
        raw_scores: list[float] = self._model.predict(pairs).tolist()

        scored = [(gc, _sigmoid(raw)) for gc, raw in zip(chunks, raw_scores)]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RankedChunk(chunk=gc.chunk, relevance_score=score, rank=i + 1)
            for i, (gc, score) in enumerate(scored[:top_k])
        ]
