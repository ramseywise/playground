"""Insights domain expert — KPIs, revenue analytics, and AR reporting."""

from __future__ import annotations

import os
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams
from google.genai import types

from schema import AssistantResponse
from .shared_tools import THINKING_CONFIG, report_out_of_domain

_INSTRUCTION = (Path(__file__).parent.parent / "prompts" / "insights_agent.txt").read_text()

_BILLY_MCP_URL = os.getenv("BILLY_MCP_URL", "http://localhost:8765/sse")

insights_agent = Agent(
    model="gemini-2.5-flash",
    name="insights_agent",
    description=(
        "Answers analytics and KPI questions: revenue summaries, invoice status, monthly trends, "
        "top customers, AR aging, DSO stats, product revenue, net margin, customer concentration, "
        "break-even analysis, and anomaly detection."
    ),
    static_instruction=types.Content(role="user", parts=[types.Part(text=_INSTRUCTION)]),
    output_schema=AssistantResponse,
    output_key="response",
    tools=[
        report_out_of_domain,
        MCPToolset(
            connection_params=SseConnectionParams(url=_BILLY_MCP_URL),
            tool_filter=[
                "get_insight_revenue_summary",
                "get_insight_invoice_status",
                "get_insight_monthly_revenue",
                "get_insight_top_customers",
                "get_insight_aging_report",
                "get_insight_customer_summary",
                "get_insight_product_revenue",
                "get_invoice_lines_summary",
                "get_invoice_dso_stats",
                "get_net_margin",
                "get_margin_by_product",
                "get_customer_concentration",
                "get_dso_trend",
                "get_break_even_estimate",
                "detect_anomaly",
            ],
        ),
    ],
    generate_content_config=THINKING_CONFIG,
)
