from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.librarian.generation.prompts import get_system_prompt
from agents.librarian.schemas.chunks import RankedChunk
from agents.librarian.schemas.retrieval import Intent
from agents.librarian.schemas.state import LibrarianState
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)

_DIRECT_INTENTS = {Intent.CONVERSATIONAL.value, Intent.OUT_OF_SCOPE.value}
_CONTEXT_SEP = "\n---\n"


def build_prompt(
    state: LibrarianState,
    ranked_chunks: list[RankedChunk],
) -> tuple[str, list[Any]]:
    """Build (system_prompt, messages_for_llm) from state and ranked chunks.

    Direct intents (conversational, out_of_scope): no context injected.
    Retrieval intents: context block = top-k chunks joined with '---'.
    Preserves full conversation history from state["messages"].
    """
    intent = state.get("intent", Intent.LOOKUP.value)
    system_prompt = get_system_prompt(intent)

    history = list(state.get("messages", []))

    if intent in _DIRECT_INTENTS or not ranked_chunks:
        return system_prompt, history

    context_parts = [
        f"[Source: {rc.chunk.metadata.url}]\n{rc.chunk.text}" for rc in ranked_chunks
    ]
    context_block = _CONTEXT_SEP.join(context_parts)

    query = state.get("standalone_query") or state.get("query", "")
    grounded_message = HumanMessage(
        content=f"Use the following sources to answer the question.\n\n{context_block}\n\nQuestion: {query}"
    )

    # Replace the last human message with the grounded version, or append
    if history and isinstance(history[-1], HumanMessage):
        messages = history[:-1] + [grounded_message]
    else:
        messages = history + [grounded_message]

    return system_prompt, messages


async def call_llm(
    llm: Any,
    system: str,
    messages: list[Any],
) -> str:
    """Invoke the LLM with system prompt prepended. Returns response text."""
    full_messages = [SystemMessage(content=system)] + messages
    response = await llm.ainvoke(full_messages)
    content = response.content if hasattr(response, "content") else str(response)
    log.info("generation.llm.done", chars=len(content))
    return content


def extract_citations(ranked_chunks: list[RankedChunk]) -> list[dict]:
    """Return [{"url": ..., "title": ...}] deduplicated by URL, in rank order."""
    seen: set[str] = set()
    citations: list[dict] = []
    for rc in ranked_chunks:
        url = rc.chunk.metadata.url
        if url not in seen:
            seen.add(url)
            citations.append({"url": url, "title": rc.chunk.metadata.title})
    return citations
