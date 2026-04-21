"""Accounting domain expert — VAT reporting, audit readiness, and P&L handoff."""

from __future__ import annotations

import os
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams
from google.genai import types

from shared.schema import AssistantResponse
from .shared_tools import THINKING_CONFIG, report_out_of_domain

_INSTRUCTION = (Path(__file__).parent.parent / "prompts" / "accounting_agent.txt").read_text()

_BILLY_MCP_URL = os.getenv("BILLY_MCP_URL", "http://localhost:8765/sse")

accounting_agent = Agent(
    model="gemini-2.5-flash",
    name="accounting_agent",
    description=(
        "Handles accounting and VAT (moms) tasks: Danish VAT summaries by quarter, "
        "audit readiness scoring, period P&L summaries, unreconciled bank transactions, "
        "and accountant handoff document generation."
    ),
    static_instruction=types.Content(role="user", parts=[types.Part(text=_INSTRUCTION)]),
    output_schema=AssistantResponse,
    output_key="response",
    tools=[
        report_out_of_domain,
        MCPToolset(
            connection_params=SseConnectionParams(url=_BILLY_MCP_URL),
            tool_filter=[
                "get_vat_summary",
                "get_unreconciled_transactions",
                "get_audit_readiness_score",
                "get_period_summary",
                "generate_handoff_doc",
            ],
        ),
    ],
    generate_content_config=THINKING_CONFIG,
)
