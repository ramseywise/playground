"""Email domain expert — sends invoices and quotes by email."""

from __future__ import annotations

import os
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams
from google.genai import types

from shared.schema import AssistantResponse
from .shared_tools import THINKING_CONFIG, report_out_of_domain

_INSTRUCTION = (Path(__file__).parent.parent / "prompts" / "email_agent.txt").read_text()

_BILLY_MCP_URL = os.getenv("BILLY_MCP_URL", "http://localhost:8765/sse")

email_agent = Agent(
    model="gemini-2.5-flash",
    name="email_agent",
    description="Sends invoices and quotes by email to customers. Drafts professional Danish email subjects and bodies.",
    static_instruction=types.Content(role="user", parts=[types.Part(text=_INSTRUCTION)]),
    output_schema=AssistantResponse,
    output_key="response",
    tools=[
        report_out_of_domain,
        MCPToolset(
            connection_params=SseConnectionParams(url=_BILLY_MCP_URL),
            tool_filter=["send_invoice_by_email", "send_quote_by_email"],
        ),
    ],
    generate_content_config=THINKING_CONFIG,
)
