"""Expense domain expert."""

from __future__ import annotations

import os
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams
from google.genai import types

from schema import AssistantResponse
from .shared_tools import THINKING_CONFIG, report_out_of_domain

_INSTRUCTION = (
    Path(__file__).parent.parent / "prompts" / "expense_agent.txt"
).read_text()

_BILLY_MCP_URL = os.getenv("BILLY_MCP_URL", "http://localhost:8765/sse")

expense_agent = Agent(
    model="gemini-2.5-flash",
    name="expense_agent",
    description=(
        "Handles business expenses: log, view, list, and analyse spending by category "
        "and vendor. Computes gross margin by comparing revenue against expenses."
    ),
    static_instruction=types.Content(
        role="user", parts=[types.Part(text=_INSTRUCTION)]
    ),
    output_schema=AssistantResponse,
    output_key="response",
    tools=[
        report_out_of_domain,
        MCPToolset(
            connection_params=SseConnectionParams(url=_BILLY_MCP_URL),
            # TODO(2): add create_expense (need test account)
            tool_filter=[
                "list_expenses",
                "get_expense",
                "get_expense_summary",
                "get_vendor_spend",
                "get_expenses_by_category",
                "get_gross_margin",
            ],
        ),
    ],
    generate_content_config=THINKING_CONFIG,
)
