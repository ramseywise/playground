"""Customer domain expert."""

from __future__ import annotations

import os
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams
from google.genai import types

from shared.schema import AssistantResponse
from .shared_tools import THINKING_CONFIG, report_out_of_domain

_INSTRUCTION = (Path(__file__).parent.parent / "prompts" / "customer_agent.txt").read_text()

_BILLY_MCP_URL = os.getenv("BILLY_MCP_URL", "http://localhost:8765/sse")

customer_agent = Agent(
    model="gemini-2.5-flash",
    name="customer_agent",
    description="Handles customers and contacts: create, view, list, and edit. Knows CVR, address, and contact persons.",
    static_instruction=types.Content(role="user", parts=[types.Part(text=_INSTRUCTION)]),
    output_schema=AssistantResponse,
    output_key="response",
    tools=[
        report_out_of_domain,
        MCPToolset(
            connection_params=SseConnectionParams(url=_BILLY_MCP_URL),
            # TODO(2): add create_customer, edit_customer (need test account)
            tool_filter=["list_customers", "get_customer"],
        ),
    ],
    generate_content_config=THINKING_CONFIG,
)
