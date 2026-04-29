"""Support domain expert — calls hc-rag-agent for help documentation Q&A."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.genai import types

from schema import AssistantResponse
from .shared_tools import SUPPORT_THINKING_CONFIG, report_out_of_domain

_INSTRUCTION = (
    Path(__file__).parent.parent / "prompts" / "support_agent.txt"
).read_text()

_HC_RAG_URL = os.getenv("HC_RAG_AGENT_URL", "http://localhost:8002")


async def search_knowledge(query: str, tool_context: Any = None) -> str:
    """Search the Billy help documentation — returns docs + confidence for agent to synthesize answer."""
    thread_id = "adk-support"
    try:
        if tool_context is not None:
            thread_id = tool_context.state.get("session_id", "adk-support")
    except Exception:
        pass
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{_HC_RAG_URL}/api/v1/retrieval",
            json={"thread_id": thread_id, "query": query},
        )
        r.raise_for_status()

    result = r.json()
    documents = result.get("documents") or []
    confidence = result.get("confidence_score", 0.0)
    escalated = result.get("escalated", False)

    if escalated:
        return f"[No docs found with confidence. Confidence: {confidence}. Consider escalating to support.]"

    if not documents:
        return "[No relevant documentation found.]"

    doc_summary = "Found relevant documentation:\n"
    for i, doc in enumerate(documents[:5], 1):
        chunk = doc.get("chunk", {})
        text = chunk.get("text", "")[:300]
        metadata = chunk.get("metadata", {})
        url = metadata.get("url", "")
        doc_summary += f"\n{i}. {text}...\n   Source: {url}"

    doc_summary += f"\n\nConfidence: {confidence:.2f}"
    return doc_summary


support_agent = Agent(
    model="gemini-2.5-flash",
    name="support_agent",
    description=(
        "Answers questions about how Billy works by searching the official help docs. "
        "Fallback for any ambiguous or how-to requests."
    ),
    static_instruction=types.Content(
        role="user", parts=[types.Part(text=_INSTRUCTION)]
    ),
    output_schema=AssistantResponse,
    output_key="response",
    tools=[
        report_out_of_domain,
        FunctionTool(func=search_knowledge),
    ],
    generate_content_config=SUPPORT_THINKING_CONFIG,
)
