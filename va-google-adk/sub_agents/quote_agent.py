"""Quote domain expert."""

from __future__ import annotations

import os
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams
from google.genai import types

from schema import AssistantResponse
from .shared_tools import THINKING_CONFIG, report_out_of_domain

_INSTRUCTION = (Path(__file__).parent.parent / "prompts" / "quote_agent.txt").read_text()

_BILLY_MCP_URL = os.getenv("BILLY_MCP_URL", "http://localhost:8765/sse")

quote_agent = Agent(
    model="gemini-2.5-flash",
    name="quote_agent",
    description=(
        "Handles quotes: create, view, list, and convert to invoice. "
        "States: open, accepted, declined, invoiced, closed."
    ),
    static_instruction=types.Content(role="user", parts=[types.Part(text=_INSTRUCTION)]),
    output_schema=AssistantResponse,
    output_key="response",
    tools=[
        report_out_of_domain,
        MCPToolset(
            connection_params=SseConnectionParams(url=_BILLY_MCP_URL),
            # TODO(2): add create_quote, edit_quote, create_invoice_from_quote (need test account)
            tool_filter=[
                "list_quotes",
                "get_quote_conversion_stats",
                "list_customers",
                "list_products",
            ],
        ),
    ],
    generate_content_config=THINKING_CONFIG,
)
