"""QA retrieval pipeline — cached backends and async retrieve/rerank steps.

LangGraph keeps separate ``retriever`` and ``reranker`` nodes; this module owns the
shared factories and :func:`retrieve_graded_chunks` / :func:`rerank_graded_chunks`
so node code stays thin and the call sequence is easy to test.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

from core.config import (
    RAG_ENSEMBLE_SCORE_THRESHOLD,
    RERANKER_BACKEND,
    RERANKER_TOP_K,
)
from rag.embedding import LangChainEmbeddingsBridge, get_embeddings
from rag.reranker.base import Reranker
from rag.reranker.cross_encoder import CrossEncoderReranker
from rag.reranker.llm_listwise_reranker import LLMListwiseReranker
from rag.reranker.passthrough import PassthroughReranker
from rag.retrieval.ensemble import EnsembleRetriever
from rag.datastore.factory import get_local_retriever
from rag.schemas.chunks import GradedChunk, RankedChunk

# Max relevance when there are no ranked chunks (matches prior node behavior).
NO_CHUNKS_CONFIDENCE: Final[float] = 0.0


@lru_cache(maxsize=1)
def get_ensemble_retriever() -> EnsembleRetriever:
    """Local vector retriever + RRF fusion (add more retrievers here later)."""
    embedder = LangChainEmbeddingsBridge(get_embeddings())
    return EnsembleRetriever(
        [get_local_retriever()],
        embedder,
        score_threshold=RAG_ENSEMBLE_SCORE_THRESHOLD,
    )


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    if RERANKER_BACKEND == "cross_encoder":
        return CrossEncoderReranker()
    if RERANKER_BACKEND == "llm_listwise":
        return LLMListwiseReranker(top_k=RERANKER_TOP_K)
    return PassthroughReranker()


async def retrieve_graded_chunks(queries: list[str], k: int) -> list[GradedChunk]:
    """Run ensemble retrieval and return fused graded chunks."""
    ensemble = get_ensemble_retriever()
    return await ensemble.retrieve(queries, k=k)


async def rerank_graded_chunks(
    query: str,
    graded: list[GradedChunk],
    top_k: int,
) -> list[RankedChunk]:
    """Rerank graded chunks for the query."""
    reranker = get_reranker()
    return await reranker.rerank(query, graded, top_k=top_k)


__all__ = [
    "NO_CHUNKS_CONFIDENCE",
    "get_ensemble_retriever",
    "get_reranker",
    "rerank_graded_chunks",
    "retrieve_graded_chunks",
]
