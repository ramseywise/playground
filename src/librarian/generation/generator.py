from __future__ import annotations

import json
from typing import Any

from clients.llm import LLMClient
from librarian.generation.prompts import get_system_prompt
from librarian.schemas.chunks import RankedChunk
from librarian.schemas.response import Citation, RAGResponse
from librarian.schemas.retrieval import Intent
from librarian.schemas.state import LibrarianState
from core.logging import get_logger

log = get_logger(__name__)

_DIRECT_INTENTS = {Intent.CONVERSATIONAL.value, Intent.OUT_OF_SCOPE.value}
_CONTEXT_SEP = "\n---\n"

_JSON_SUFFIX = """

Respond ONLY with valid JSON matching this schema — no markdown fences, no extra text:
{
  "answer": "<your answer>",
  "citations": [{"url": "<source url>", "title": "<source title>", "snippet": "<relevant excerpt>"}],
  "confidence": "high" | "medium" | "low",
  "follow_up": "<optional follow-up question>"
}"""

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


async def call_llm_structured(
    llm: LLMClient,
    system: str,
    messages: list[dict[str, str]],
    ranked_chunks: list[RankedChunk],
) -> RAGResponse:
    """Call the LLM with a JSON-mode system prompt and parse into RAGResponse.

    Falls back to wrapping the raw text if JSON parsing fails.
    """
    structured_system = system + _JSON_SUFFIX
    raw = await llm.generate(structured_system, messages)
    log.info("generation.structured.raw", chars=len(raw))

    try:
        return RAGResponse.model_validate_json(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # Retry after stripping markdown fences
    from core.parsing.json import strip_json_fences

    stripped = strip_json_fences(raw)
    try:
        return RAGResponse.model_validate_json(stripped)
    except (json.JSONDecodeError, ValueError):
        log.warning("generation.structured.fallback", raw_preview=raw[:200])

    # Fallback: wrap raw text with citations extracted from chunks
    fallback_citations = [
        Citation(url=rc.chunk.metadata.url, title=rc.chunk.metadata.title)
        for rc in ranked_chunks
        if rc.chunk.metadata.url
    ]
    # Deduplicate by URL
    seen: set[str] = set()
    unique_citations: list[Citation] = []
    for c in fallback_citations:
        if c.url not in seen:
            seen.add(c.url)
            unique_citations.append(c)

    return RAGResponse(
        answer=raw,
        citations=unique_citations,
        confidence="low",
    )


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
