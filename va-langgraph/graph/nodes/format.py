"""Format node — wraps domain tool results into a final AssistantResponse.

Runs after the domain subgraph.  Calls the LLM once with the tool results
and the original user message to produce structured AssistantResponse JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from model_factory import resolve_chat_model
from schema import AssistantResponse
from ..state import AgentState

log = structlog.get_logger(__name__)

_SYSTEM = (Path(__file__).parent.parent.parent / "prompts" / "format.txt").read_text()


def _get_structured_llm():
    return resolve_chat_model("medium").with_structured_output(AssistantResponse)


async def format_node(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    tool_results = state.get("tool_results", [])
    intent = state.get("intent", "support")
    page_url = state.get("page_url")

    user_text = messages[-1].content if messages else "(no message)"
    if page_url:
        user_text = f"[User is on page: {page_url}]\n{user_text}"

    tool_summary = (
        json.dumps(tool_results, ensure_ascii=False, indent=2)
        if tool_results
        else "(no tool calls made)"
    )

    prompt = f"""User request: {user_text}

Intent: {intent}

Tool results:
{tool_summary}

Produce the AssistantResponse JSON."""

    try:
        result = await _get_structured_llm().ainvoke(
            [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        response_dict = result.model_dump()
    except Exception as e:
        log.exception("format_node.failed", error=str(e))
        # Fallback: plain text from tool results
        response_dict = AssistantResponse(
            message=f"Done. {tool_summary[:500]}"
        ).model_dump()

    return {**state, "response": response_dict}
