"""Option 3: ADK Agent with custom RAG tools.

The LLM (Gemini) decides *when* to search, *when* to rerank, and *whether*
to retry — unlike the LangGraph pipeline which follows a fixed CRAG loop.

This tests the hypothesis that an LLM-driven retrieval strategy may make
better decisions for some query types (e.g. exploratory, multi-hop) while
potentially under-retrieving for straightforward lookups.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents import Agent

from orchestration.adk.tools import (
    configure_tools,
    rerank_results,
    search_knowledge_base,
)
from librarian.config import LibrarySettings
from librarian.reranker.base import Reranker
from librarian.retrieval.base import Embedder, Retriever
from core.logging import get_logger

log = get_logger(__name__)

# Gemini 2.0 Flash for tool-calling
_DEFAULT_MODEL = "gemini-2.0-flash"

_INSTRUCTION = """\
You are a knowledgeable research assistant with access to a curated knowledge base.

When answering questions:
1. ALWAYS use search_knowledge_base first to find relevant passages.
2. Review the search results. If they seem noisy or you need higher precision, \
use rerank_results to surface the best matches.
3. If the top reranked result has confidence below 0.3, try searching again \
with different phrasing or broader terms.
4. Base your answer strictly on retrieved passages — do not hallucinate or speculate.
5. Cite the source URL inline when referencing specific facts (e.g. [title](url)).
6. If no relevant passages are found after searching, say so clearly — \
do not make up an answer.
7. Keep answers concise and directly responsive to the question.
"""


def create_custom_rag_agent(
    cfg: LibrarySettings,
    *,
    retriever: Retriever,
    embedder: Embedder,
    reranker: Reranker,
    model: str = _DEFAULT_MODEL,
) -> Agent:
    """Build an ADK Agent with custom RAG tools.

    The agent uses Gemini for orchestration (tool-calling decisions)
    while the actual retrieval and reranking use the same Librarian
    infrastructure as the LangGraph pipeline.

    Args:
        cfg: Library settings (used for logging / config context).
        retriever: The vector store retriever (Chroma, OpenSearch, etc.).
        embedder: The embedding model (E5, MiniLM, etc.).
        reranker: The reranking model (cross-encoder, LLM-listwise, etc.).
        model: Gemini model name for the orchestrating agent.

    Returns:
        A configured ADK Agent ready for ``Runner.run_async()``.
    """
    # Wire the retrieval components into the tool functions
    configure_tools(retriever=retriever, embedder=embedder, reranker=reranker)

    agent = Agent(
        model=model,
        name="custom_rag",
        description=(
            "LLM-driven RAG: the model decides when and how to retrieve, "
            "rerank, and retry. Uses the same retrieval stack as the Librarian "
            "pipeline but with Gemini controlling the orchestration."
        ),
        instruction=_INSTRUCTION,
        tools=[search_knowledge_base, rerank_results],
    )

    log.info(
        "adk.custom_rag.created",
        model=model,
        retrieval_strategy=cfg.retrieval_strategy,
        reranker_strategy=cfg.reranker_strategy,
    )

    return agent


async def run_custom_rag_query(
    agent: Agent,
    query: str,
    *,
    session_id: str = "default",
    user_id: str = "eval",
) -> dict[str, Any]:
    """Run a single query through the custom RAG agent and return the response.

    This is a convenience wrapper for eval and testing — it handles
    ADK session setup, runner creation, and response extraction.

    Returns:
        Dict with 'response' (str) and 'events' (list of ADK events).
    """
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="librarian_eval",
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name="librarian_eval",
        user_id=user_id,
    )

    user_content = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )

    events = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=user_content,
    ):
        events.append(event)

    # Extract final response — the last agent event with text content
    response_text = ""
    for event in reversed(events):
        if event.author == agent.name and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    response_text = part.text
                    break
            if response_text:
                break

    return {
        "response": response_text,
        "events": events,
        "session_id": session.id,
    }
