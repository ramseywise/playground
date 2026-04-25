"""Format node — wraps domain tool results into a final AssistantResponse.

Runs after the domain subgraph.  Calls the LLM once with the tool results
and the original user message to produce structured AssistantResponse JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from schema import AssistantResponse
from ..state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = (Path(__file__).parent.parent.parent / "prompts" / "format.txt").read_text()

def _get_structured_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return llm.with_structured_output(AssistantResponse)


async def format_node(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    tool_results = state.get("tool_results", [])
    intent = state.get("intent", "support")
    page_url = state.get("page_url")

    user_text = messages[-1].content if messages else "(no message)"
    if page_url:
        user_text = f"[User is on page: {page_url}]\n{user_text}"

    tool_summary = json.dumps(tool_results, ensure_ascii=False, indent=2) if tool_results else "(no tool calls made)"

    prompt = f"""User request: {user_text}

Intent: {intent}

Tool results:
{tool_summary}

Produce the AssistantResponse JSON."""

    try:
        result = await _get_structured_llm().ainvoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])
        response_dict = result.model_dump()
    except Exception as e:
        logger.exception("format_node structured output failed: %s", e)
        # Fallback: plain text from tool results
        response_dict = AssistantResponse(
            message=f"Done. {tool_summary[:500]}"
        ).model_dump()

    return {**state, "response": response_dict}
