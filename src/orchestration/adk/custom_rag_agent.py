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

from orchestration.adk.callbacks import (
    after_agent,
    after_tool,
    before_agent,
    before_tool,
)
from orchestration.adk.tools import (
    analyze_query,
    condense_query,
    configure_tools,
    escalate,
    rerank_results,
    search_knowledge_base,
)
from librarian.config import LibrarySettings
from librarian.reranker.base import Reranker
from librarian.retrieval.base import Embedder, Retriever
from core.clients.llm import LLMClient
from core.logging import get_logger

log = get_logger(__name__)

# Gemini 2.0 Flash for tool-calling
_DEFAULT_MODEL = "gemini-2.0-flash"

_INSTRUCTION = """\
You are a knowledgeable research assistant with access to a curated knowledge base.

Follow this workflow when answering questions:

**Step 1 — Understand the query:**
- If the conversation has prior messages, use condense_query to rewrite the \
user's latest message as a standalone question.
- Use analyze_query to understand the intent, extract entities, and get \
recommended search terms and retrieval strategy.
- **If intent is "out_of_scope"**, use escalate immediately with reason \
"out_of_scope". Do NOT attempt to answer out-of-scope questions.
- If intent is "conversational", respond directly without searching.

**Step 2 — Retrieve information:**
- Use search_knowledge_base with the standalone query (or original if single-turn).
- If analyze_query suggested expanded_terms, consider searching with those too.
- For complex queries with sub_queries, search each sub-question separately.

**Step 3 — Refine results:**
- Review search results. If they seem noisy or low-quality, use rerank_results \
to surface the best matches.
- If the top reranked result has confidence below 0.3, try searching again \
with different phrasing or broader terms from the analysis.
- **If confidence is still below 0.2 after retrying**, use escalate with reason \
"low_confidence" — don't guess.

**Step 4 — Answer:**
- Base your answer strictly on retrieved passages — do not hallucinate or speculate.
- Cite the source URL inline when referencing specific facts (e.g. [title](url)).
- Keep answers concise and directly responsive to the question.
- **If the user asks to speak to a human**, use escalate with reason \
"explicit_request".
"""


def create_custom_rag_agent(
    cfg: LibrarySettings,
    *,
    retriever: Retriever,
    embedder: Embedder,
    reranker: Reranker,
    condenser_llm: LLMClient | None = None,
    model: str = _DEFAULT_MODEL,
) -> Agent:
    """Build an ADK Agent with the full custom RAG tool suite.

    The agent uses Gemini for orchestration (tool-calling decisions)
    while the actual retrieval, reranking, query understanding, and
    condensation use the same Librarian infrastructure as the LangGraph
    pipeline.

    Args:
        cfg: Library settings (used for logging / config context).
        retriever: The vector store retriever (Chroma, OpenSearch, etc.).
        embedder: The embedding model (E5, MiniLM, etc.).
        reranker: The reranking model (cross-encoder, LLM-listwise, etc.).
        condenser_llm: LLM for multi-turn query rewriting (e.g. Haiku). Optional.
        model: Gemini model name for the orchestrating agent.

    Returns:
        A configured ADK Agent ready for ``Runner.run_async()``.
    """
    configure_tools(
        retriever=retriever,
        embedder=embedder,
        reranker=reranker,
        condenser_llm=condenser_llm,
    )

    agent = Agent(
        model=model,
        name="custom_rag",
        description=(
            "LLM-driven RAG: the model decides when and how to retrieve, "
            "rerank, and retry. Uses the same retrieval stack as the Librarian "
            "pipeline but with Gemini controlling the orchestration."
        ),
        instruction=_INSTRUCTION,
        tools=[
            analyze_query,
            condense_query,
            search_knowledge_base,
            rerank_results,
            escalate,
        ],
        before_agent_callback=before_agent,
        after_agent_callback=after_agent,
        before_tool_callback=before_tool,
        after_tool_callback=after_tool,
    )

    log.info(
        "adk.custom_rag.created",
        model=model,
        tool_count=len(agent.tools),
        retrieval_strategy=cfg.retrieval_strategy,
        reranker_strategy=cfg.reranker_strategy,
        has_condenser=condenser_llm is not None,
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
