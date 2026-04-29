"""Banking and cashflow domain expert."""

from __future__ import annotations

import os
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams
from google.genai import types

from schema import AssistantResponse
from .shared_tools import THINKING_CONFIG, report_out_of_domain

_INSTRUCTION = (
    Path(__file__).parent.parent / "prompts" / "banking_agent.txt"
).read_text()

_BILLY_MCP_URL = os.getenv("BILLY_MCP_URL", "http://localhost:8765/sse")

banking_agent = Agent(
    model="gemini-2.5-flash",
    name="banking_agent",
    description=(
        "Handles banking and cashflow: current account balances, bank transactions, "
        "reconciling payments to invoices, cashflow forecasting, and runway estimation."
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
            tool_filter=[
                "get_bank_balance",
                "list_bank_transactions",
                "match_transaction_to_invoice",
                "get_cashflow_forecast",
                "get_runway_estimate",
            ],
        ),
    ],
    generate_content_config=THINKING_CONFIG,
)
