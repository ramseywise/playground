from __future__ import annotations

from typing import Any

from core.clients.llm import LLMClient
from librarian.schemas.state import LibrarianState
from core.logging import get_logger

log = get_logger(__name__)

_SYSTEM_PROMPT = (
    "Rewrite the user's latest message as a standalone query using the full "
    "conversation context. Return only the rewritten query."
)


def _message_to_dict(message: object) -> dict[str, str]:  # noqa: D103
    if isinstance(message, dict):
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        return {"role": role, "content": content}
    role = getattr(message, "type", "user")
    content = getattr(message, "content", str(message))
    return {"role": "assistant" if role == "ai" else "user", "content": str(content)}


class CondenserAgent:
    """Rewrite multi-turn queries into standalone questions.

    On single-turn conversations (one message or fewer) this is a no-op.
    On multi-turn it calls the LLM (Haiku) to resolve coreferences and produce
    a self-contained query for downstream retrieval.
    """

    name = "condenser"
    description = "Rewrite multi-turn queries into standalone questions"
    instruction = _SYSTEM_PROMPT

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def condense(self, state: LibrarianState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        query = state.get("query", "")
        if len(messages) <= 1:
            return {"standalone_query": query}

        payload = [_message_to_dict(message) for message in messages]
        standalone_query = await self._llm.generate(_SYSTEM_PROMPT, payload)
        standalone_query = standalone_query.strip() or query

        log.info(
            "history.condense.done",
            message_count=len(messages),
            standalone_query=standalone_query[:120],
        )
        return {"standalone_query": standalone_query}

    def as_node(self) -> Any:
        """Return a LangGraph-compatible async node function."""
        async def _condense_node(state: LibrarianState) -> dict[str, Any]:
            return await self.condense(state)

        return _condense_node
