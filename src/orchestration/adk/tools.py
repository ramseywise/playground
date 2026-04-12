"""ADK FunctionTools wrapping the Librarian retrieval, reranking, and query understanding stack.

These are plain async Python functions with type hints and docstrings.
ADK auto-wraps them as ``FunctionTool`` when passed to an ``Agent``.

Components are injected via ``configure_tools()`` at startup into a
module-level ``ToolDeps`` container (not bare globals).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.clients.llm import LLMClient
from librarian.plan.analyzer import QueryAnalyzer
from librarian.reranker.base import Reranker
from librarian.retrieval.base import Embedder, Retriever
from librarian.schemas.chunks import GradedChunk
from core.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dependency container — set via configure_tools()
# ---------------------------------------------------------------------------


@dataclass
class ToolDeps:
    """Injected dependencies for ADK tool functions.

    Using a container instead of bare module globals for testability
    and thread-safety.
    """

    retriever: Retriever
    embedder: Embedder
    reranker: Reranker
    condenser_llm: LLMClient | None = None
    analyzer: QueryAnalyzer | None = None


_deps: ToolDeps | None = None


def configure_tools(
    retriever: Retriever,
    embedder: Embedder,
    reranker: Reranker,
    *,
    condenser_llm: LLMClient | None = None,
    analyzer: QueryAnalyzer | None = None,
) -> ToolDeps:
    """Inject retrieval components into the tool functions.

    Must be called before any tool function is invoked (typically at
    application startup or test setup).

    Returns the ``ToolDeps`` container for direct use if needed.
    """
    global _deps  # noqa: PLW0603
    _deps = ToolDeps(
        retriever=retriever,
        embedder=embedder,
        reranker=reranker,
        condenser_llm=condenser_llm,
        analyzer=analyzer or QueryAnalyzer(),
    )
    log.info("adk.tools.configured")
    return _deps


def _get_deps() -> ToolDeps:
    """Return the configured ToolDeps or raise."""
    if _deps is None:
        msg = (
            "ADK tools not configured — call configure_tools() before using "
            "any tool function"
        )
        raise RuntimeError(msg)
    return _deps


# Keep backward-compat alias for existing tests
_check_configured = _get_deps


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
    deps = _get_deps()

    log.info("adk.tool.search", query=query[:80], k=num_results)

    query_vector = await deps.embedder.aembed_query(query)
    raw_results = await deps.retriever.search(
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
    passages: list[dict[str, Any]],
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
    deps = _get_deps()

    log.info("adk.tool.rerank", query=query[:80], n_passages=len(passages), top_k=top_k)

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

    ranked = await deps.reranker.rerank(query, graded_chunks, top_k=top_k)

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


# ---------------------------------------------------------------------------
# Tool: analyze_query
# ---------------------------------------------------------------------------


def analyze_query(query: str) -> dict[str, Any]:
    """Analyze a query to understand its intent, entities, complexity, and best retrieval strategy.

    Use this BEFORE searching to understand what the user is asking and how
    best to retrieve relevant information. The analysis includes:
    - Intent classification (lookup, comparison, exploration, etc.)
    - Named entity extraction (technologies, concepts, people)
    - Sub-query decomposition for complex questions
    - Term expansion with domain synonyms
    - Recommended retrieval mode (dense, hybrid, or snippet)

    Args:
        query: The user's question in natural language.

    Returns:
        A dict with:
        - intent: the classified intent (lookup, compare, explore, etc.)
        - confidence: how confident the classification is (0-1)
        - entities: dict of entity_type → list of matched strings
        - sub_queries: list of decomposed sub-questions (for complex queries)
        - expanded_terms: list of related search terms and synonyms
        - complexity: "simple", "moderate", or "complex"
        - retrieval_mode: recommended mode — "dense", "hybrid", or "snippet"
    """
    deps = _get_deps()
    analyzer = deps.analyzer or QueryAnalyzer()

    log.info("adk.tool.analyze", query=query[:80])

    analysis = analyzer.analyze(query)

    result = {
        "intent": analysis.intent.value,
        "confidence": round(analysis.confidence, 3),
        "entities": analysis.entities,
        "sub_queries": analysis.sub_queries,
        "expanded_terms": analysis.expanded_terms,
        "complexity": analysis.complexity,
        "retrieval_mode": analysis.retrieval_mode,
    }

    log.info(
        "adk.tool.analyze.done",
        intent=result["intent"],
        complexity=result["complexity"],
        entity_count=sum(len(v) for v in analysis.entities.values()),
        expansion_count=len(analysis.expanded_terms),
    )

    return result


# ---------------------------------------------------------------------------
# Tool: condense_query
# ---------------------------------------------------------------------------


async def condense_query(
    query: str,
    conversation_history: list[dict[str, str]],
) -> dict[str, Any]:
    """Rewrite a follow-up question into a standalone query using conversation context.

    Use this when the user's message references previous conversation context
    (e.g. "what about that one?", "how does it compare?", "tell me more").
    The tool uses a lightweight LLM to resolve coreferences and produce a
    self-contained query that can be used for search.

    Args:
        query: The user's latest message (may contain coreferences).
        conversation_history: List of prior messages, each with "role" and "content" keys.
            Example: [{"role": "user", "content": "what is OAuth?"}, {"role": "assistant", "content": "OAuth is..."}]

    Returns:
        A dict with:
        - standalone_query: the rewritten self-contained query
        - was_rewritten: whether the query was actually changed
    """
    deps = _get_deps()

    # Single-turn or no history — pass through unchanged
    if not conversation_history or len(conversation_history) <= 1:
        return {"standalone_query": query, "was_rewritten": False}

    if deps.condenser_llm is None:
        log.warning(
            "adk.tool.condense.no_llm",
            msg="condenser_llm not configured, passing through",
        )
        return {"standalone_query": query, "was_rewritten": False}

    log.info(
        "adk.tool.condense", query=query[:80], history_len=len(conversation_history)
    )

    system_prompt = (
        "Rewrite the user's latest message as a standalone query using the full "
        "conversation context. Return only the rewritten query."
    )

    standalone = await deps.condenser_llm.generate(system_prompt, conversation_history)
    standalone = standalone.strip() or query
    was_rewritten = standalone != query

    log.info(
        "adk.tool.condense.done",
        original=query[:80],
        standalone=standalone[:80],
        was_rewritten=was_rewritten,
    )

    return {
        "standalone_query": standalone,
        "was_rewritten": was_rewritten,
    }


# ---------------------------------------------------------------------------
# Tool: escalate
# ---------------------------------------------------------------------------

_ESCALATION_REASONS = {
    "out_of_scope": (
        "This question falls outside the knowledge base. "
        "I can only answer questions about the curated document corpus."
    ),
    "low_confidence": (
        "I wasn't able to find sufficiently relevant information to give "
        "a reliable answer. A human reviewer may be able to help."
    ),
    "sensitive_topic": (
        "This question may require human judgment. "
        "I'm escalating to a human reviewer for a more considered response."
    ),
    "explicit_request": ("Understood — connecting you with a human reviewer."),
}


def escalate(
    reason: str,
    query: str,
    context: str = "",
) -> dict[str, Any]:
    """Escalate a query to a human agent when the assistant cannot or should not answer.

    Use this tool when:
    - analyze_query returns intent "out_of_scope" (weather, sports, recipes, etc.)
    - Search results have very low confidence after reranking (confidence < 0.2)
    - The user explicitly asks to speak to a human
    - The topic is sensitive and requires human judgment

    This does NOT answer the question — it signals that a human should handle it.

    Args:
        reason: Why the escalation is needed. One of:
            "out_of_scope" — query is outside the knowledge base domain
            "low_confidence" — retrieval didn't find relevant information
            "sensitive_topic" — requires human judgment
            "explicit_request" — user asked for a human
        query: The original user query being escalated.
        context: Optional additional context for the human reviewer
            (e.g. what was searched, what was found).

    Returns:
        A dict with:
        - escalated: always True
        - reason: the escalation reason
        - message: a user-facing message explaining the escalation
        - reviewer_context: context for the human reviewer
    """
    log.info(
        "adk.tool.escalate",
        reason=reason,
        query=query[:80],
        has_context=bool(context),
    )

    message = _ESCALATION_REASONS.get(reason, _ESCALATION_REASONS["out_of_scope"])

    return {
        "escalated": True,
        "reason": reason,
        "message": message,
        "reviewer_context": {
            "query": query,
            "reason": reason,
            "context": context,
        },
    }
