"""Multi-agent coordinator — routes queries to the best sub-agent.

Uses Gemini to classify the query type and delegate to either:
- LibrarianADKAgent (knowledge/lookup/comparison queries → full CRAG pipeline)
- CustomRAGAgent (exploratory/open-ended queries → LLM-driven tool-calling)

This tests ADK's multi-agent composition — whether a routing layer
adds value over a single pipeline.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents import Agent

from core.logging import get_logger

log = get_logger(__name__)

_COORDINATOR_MODEL = "gemini-2.0-flash"

_COORDINATOR_INSTRUCTION = """\
You are a query router. Analyze the user's question and delegate to the
best sub-agent:

- **librarian_hybrid**: Use for knowledge lookups, factual questions,
  comparisons, and questions that need precise citations from the corpus.
  This agent has a thorough multi-step retrieval pipeline.

- **custom_rag**: Use for exploratory questions, brainstorming, multi-hop
  reasoning, or questions where the user wants a broader search.
  This agent can adaptively search and re-search.

Route to librarian_hybrid by default. Only route to custom_rag when the
question is clearly exploratory or open-ended.
"""


def create_coordinator(
    librarian_agent: Agent,
    custom_rag_agent: Agent,
    *,
    model: str = _COORDINATOR_MODEL,
) -> Agent:
    """Build a coordinator agent that routes between sub-agents.

    Args:
        librarian_agent: The LibrarianADKAgent (Option 4 hybrid).
        custom_rag_agent: The CustomRAGAgent (Option 3).
        model: Gemini model for the coordinator's routing decisions.

    Returns:
        An ADK Agent with sub_agents configured.
    """
    coordinator = Agent(
        model=model,
        name="coordinator",
        description="Routes queries to the best sub-agent based on query type",
        instruction=_COORDINATOR_INSTRUCTION,
        sub_agents=[librarian_agent, custom_rag_agent],
    )

    log.info(
        "adk.coordinator.created",
        model=model,
        sub_agents=[librarian_agent.name, custom_rag_agent.name],
    )

    return coordinator
