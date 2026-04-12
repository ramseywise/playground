"""ADK FunctionTools wrapping the Librarian retrieval and reranking stack.

These are plain async Python functions with type hints and docstrings.
ADK auto-wraps them as ``FunctionTool`` when passed to an ``Agent``.

The tools use the same Retriever/Embedder/Reranker infrastructure as the
LangGraph pipeline — they just expose it as LLM-callable functions so
the ADK agent can decide *when* to search and *when* to rerank.

Components are injected via ``configure_tools()`` at startup.
"""

from __future__ import annotations

from typing import Any

from librarian.reranker.base import Reranker
from librarian.retrieval.base import Embedder, Retriever
from librarian.schemas.chunks import GradedChunk
from core.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level singletons — set via configure_tools()
# ---------------------------------------------------------------------------

_retriever: Retriever | None = None
_embedder: Embedder | None = None
_reranker: Reranker | None = None


def configure_tools(
    retriever: Retriever,
    embedder: Embedder,
    reranker: Reranker,
) -> None:
    """Inject retrieval components into the tool functions.

    Must be called before any tool function is invoked (typically at
    application startup or test setup).
    """
    global _retriever, _embedder, _reranker  # noqa: PLW0603
    _retriever = retriever
    _embedder = embedder
    _reranker = reranker
    log.info("adk.tools.configured")


def _check_configured() -> tuple[Retriever, Embedder, Reranker]:
    """Raise if tools have not been configured."""
    if _retriever is None or _embedder is None or _reranker is None:
        msg = (
            "ADK tools not configured — call configure_tools() before using "
            "search_knowledge_base or rerank_results"
        )
        raise RuntimeError(msg)
    return _retriever, _embedder, _reranker


# ---------------------------------------------------------------------------
# Tool: search_knowledge_base
# ---------------------------------------------------------------------------


async def search_knowledge_base(
    query: str,
    num_results: int = 10,
) -> dict[str, Any]:
    """Search the knowledge base for passages relevant to the query.

    Uses hybrid search (vector similarity + BM25 keyword matching) over
    the curated document corpus. Returns ranked passages with metadata.

    Args:
        query: The search query — a natural language question or topic.
        num_results: Maximum number of passages to return (default 10).

    Returns:
        A dict with:
        - results: list of passages with text, url, title, and score
        - total: number of results returned
    """
    retriever, embedder, _ = _check_configured()

    log.info("adk.tool.search", query=query[:80], k=num_results)

    query_vector = await embedder.aembed_query(query)
    raw_results = await retriever.search(
        query_text=query,
        query_vector=query_vector,
        k=num_results,
    )

    results = [
        {
            "text": r.chunk.text,
            "url": r.chunk.metadata.url,
            "title": r.chunk.metadata.title,
            "score": round(r.score, 4),
            "chunk_id": r.chunk.id,
        }
        for r in raw_results
    ]

    log.info("adk.tool.search.done", query=query[:80], result_count=len(results))

    return {
        "results": results,
        "total": len(results),
    }


# ---------------------------------------------------------------------------
# Tool: rerank_results
# ---------------------------------------------------------------------------


async def rerank_results(
    query: str,
    passages: list[dict[str, str]],
    top_k: int = 3,
) -> dict[str, Any]:
    """Re-rank passages by relevance to the query using a cross-encoder model.

    Takes passages from search_knowledge_base and re-scores them with a
    more accurate (but slower) cross-encoder model. Use this when initial
    search results seem noisy or you need high-precision answers.

    Args:
        query: The original user question.
        passages: List of passage dicts, each with at least "text" and "chunk_id" keys.
        top_k: Number of top passages to return after reranking (default 3).

    Returns:
        A dict with:
        - results: list of reranked passages with relevance_score and rank
        - confidence: maximum relevance_score (0-1) — indicates overall quality
    """
    _, _, reranker = _check_configured()

    log.info("adk.tool.rerank", query=query[:80], n_passages=len(passages), top_k=top_k)

    # Convert passage dicts to GradedChunks for the reranker protocol
    from librarian.schemas.chunks import Chunk, ChunkMetadata

    graded_chunks = []
    for p in passages:
        chunk = Chunk(
            id=p.get("chunk_id", "unknown"),
            text=p.get("text", ""),
            metadata=ChunkMetadata(
                url=p.get("url", ""),
                title=p.get("title", ""),
                doc_id=p.get("chunk_id", "unknown"),
            ),
        )
        graded_chunks.append(
            GradedChunk(chunk=chunk, score=float(p.get("score", 0.5)), relevant=True)
        )

    ranked = await reranker.rerank(query, graded_chunks, top_k=top_k)

    results = [
        {
            "text": r.chunk.text,
            "url": r.chunk.metadata.url,
            "title": r.chunk.metadata.title,
            "relevance_score": round(r.relevance_score, 4),
            "rank": r.rank,
            "chunk_id": r.chunk.id,
        }
        for r in ranked
    ]

    confidence = max((r.relevance_score for r in ranked), default=0.0)

    log.info(
        "adk.tool.rerank.done",
        query=query[:80],
        result_count=len(results),
        confidence=round(confidence, 4),
    )

    return {
        "results": results,
        "confidence": round(confidence, 4),
    }
