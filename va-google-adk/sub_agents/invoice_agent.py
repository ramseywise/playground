"""Invoice domain expert."""

from __future__ import annotations

import os
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams
from google.genai import types

from schema import AssistantResponse
from .shared_tools import THINKING_CONFIG, report_out_of_domain

_INSTRUCTION = (Path(__file__).parent.parent / "prompts" / "invoice_agent.txt").read_text()

_BILLY_MCP_URL = os.getenv("BILLY_MCP_URL", "http://localhost:8765/sse")

invoice_agent = Agent(
    model="gemini-2.5-flash",
    name="invoice_agent",
    description=(
        "Handles invoices: create, view, list, edit, approve, and summarize. "
        "Covers DKK amounts, VAT, payment terms, and draft vs approved states."
    ),
    static_instruction=types.Content(role="user", parts=[types.Part(text=_INSTRUCTION)]),
    output_schema=AssistantResponse,
    output_key="response",
    tools=[
        report_out_of_domain,
        MCPToolset(
            connection_params=SseConnectionParams(url=_BILLY_MCP_URL),
            # TODO(2): add create_invoice, edit_invoice, void_invoice, send_invoice_reminder (need test account)
            tool_filter=[
                "get_invoice",
                "list_invoices",
                "get_invoice_summary",
                "get_invoice_dso_stats",
                "list_customers",
                "list_products",
            ],
        ),
    ],
    generate_content_config=THINKING_CONFIG,
)
