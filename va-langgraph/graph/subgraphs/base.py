"""Base class for domain subgraph nodes.

Each domain subgraph is a single async function that:
  1. Receives AgentState
  2. Calls the LLM with its domain tools (ReAct loop via create_react_agent)
  3. Accumulates tool_results in state
  4. Returns updated state

The format_node then converts tool_results → AssistantResponse.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

from ..state import AgentState

logger = logging.getLogger(__name__)


async def run_domain(
    state: AgentState,
    system_prompt: str,
    tools: list[BaseTool],
    model: str = "gemini-2.5-flash",
) -> AgentState:
    """Run a ReAct loop for a domain agent.

    Uses bind_tools for tool calling. Loops until the model stops
    calling tools or hits a max-iterations guard.
    """
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    messages = state.get("messages", [])
    page_url = state.get("page_url")
    user_text = messages[-1].content if messages else ""
    if page_url:
        user_text = f"[User is on page: {page_url}]\n{user_text}"

    loop_messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=str(user_text)),
    ]

    tool_results: list[dict] = list(state.get("tool_results", []))
    max_iterations = 8

    for _ in range(max_iterations):
        response: AIMessage = await llm_with_tools.ainvoke(loop_messages)
        loop_messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            tool = tools_by_name.get(tool_name)
            if tool is None:
                result = f"Unknown tool: {tool_name}"
            else:
                try:
                    if hasattr(tool, "ainvoke"):
                        result = await tool.ainvoke(tool_args)
                    else:
                        result = tool.invoke(tool_args)
                except Exception as e:
                    logger.exception("Tool %s failed", tool_name)
                    result = {"error": str(e)}

            tool_results.append({"tool": tool_name, "args": tool_args, "result": result})
            loop_messages.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=tool_id))

    return {**state, "tool_results": tool_results}
