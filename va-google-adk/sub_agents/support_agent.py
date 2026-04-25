"""Support domain expert — searches the Billy help documentation."""

from __future__ import annotations

import os
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams
from google.genai import types

from schema import AssistantResponse
from .shared_tools import SUPPORT_THINKING_CONFIG, report_out_of_domain

_INSTRUCTION = (Path(__file__).parent.parent / "prompts" / "support_agent.txt").read_text()

_BILLY_MCP_URL = os.getenv("BILLY_MCP_URL", "http://localhost:8765/sse")

support_agent = Agent(
    model="gemini-2.5-flash",
    name="support_agent",
    description=(
        "Answers questions about how Billy works by searching the official help docs. "
        "Fallback for any ambiguous or how-to requests."
    ),
    static_instruction=types.Content(role="user", parts=[types.Part(text=_INSTRUCTION)]),
    output_schema=AssistantResponse,
    output_key="response",
    tools=[
        report_out_of_domain,
        MCPToolset(
            connection_params=SseConnectionParams(url=_BILLY_MCP_URL),
            tool_filter=["fetch_support_knowledge"],
        ),
    ],
    generate_content_config=SUPPORT_THINKING_CONFIG,
)
