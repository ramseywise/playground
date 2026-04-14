from __future__ import annotations

from typing import Any

from clients.llm import LLMClient
from librarian.generation.prompts import get_system_prompt
from librarian.schemas.chunks import RankedChunk
from librarian.schemas.retrieval import Intent
from librarian.schemas.state import LibrarianState
from core.logging import get_logger

log = get_logger(__name__)

_DIRECT_INTENTS = {Intent.CONVERSATIONAL.value, Intent.OUT_OF_SCOPE.value}
_CONTEXT_SEP = "\n---\n"

# LangGraph BaseMessage.type → anthropic API role mapping
_ROLE_MAP: dict[str, str] = {"human": "user", "ai": "assistant"}


def _message_to_dict(msg: object) -> dict[str, str]:
    """Convert a LangGraph BaseMessage to an anthropic API message dict.

    Handles both BaseMessage objects (from langgraph state) and plain dicts.
    """
    if isinstance(msg, dict):
        return msg
    role = _ROLE_MAP.get(getattr(msg, "type", ""), "user")
    content = msg.content if hasattr(msg, "content") else str(msg)
    return {"role": role, "content": content}


def build_prompt(
    state: LibrarianState,
    ranked_chunks: list[RankedChunk],
) -> tuple[str, list[dict[str, str]]]:
    """Build (system_prompt, messages_for_llm) from state and ranked chunks.

    Returns messages as plain dicts ``{"role": ..., "content": ...}`` suitable
    for the anthropic messages API.  BaseMessage objects from langgraph state
    are converted at this boundary.

    Direct intents (conversational, out_of_scope): no context injected.
    Retrieval intents: context block = top-k chunks joined with '---'.
    Preserves full conversation history from state["messages"].
    """
    intent = state.get("intent", Intent.LOOKUP.value)
    system_prompt = get_system_prompt(intent)

    history = [_message_to_dict(m) for m in state.get("messages", [])]

    if intent in _DIRECT_INTENTS or not ranked_chunks:
        return system_prompt, history

    context_parts = [
        f"[Source: {rc.chunk.metadata.url}]\n{rc.chunk.text}" for rc in ranked_chunks
    ]
    context_block = _CONTEXT_SEP.join(context_parts)

    query = state.get("standalone_query") or state.get("query", "")
    grounded_message: dict[str, str] = {
        "role": "user",
        "content": f"Use the following sources to answer the question.\n\n{context_block}\n\nQuestion: {query}",
    }

    # Replace the last human message with the grounded version, or append
    if history and history[-1].get("role") == "user":
        messages = history[:-1] + [grounded_message]
    else:
        messages = history + [grounded_message]

    return system_prompt, messages


async def call_llm(
    llm: LLMClient,
    system: str,
    messages: list[dict[str, str]],
) -> str:
    """Invoke the LLM and return the response text.

    Expects *llm* to implement ``generate(system, messages) -> str``
    (i.e. ``AnthropicLLM`` or a mock with the same interface).
    """
    content = await llm.generate(system, messages)
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
