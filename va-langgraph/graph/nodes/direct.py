"""Direct node — handles greetings and fully out-of-scope requests inline."""

from __future__ import annotations

from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI

from schema import AssistantResponse
from ..state import AgentState

_SYSTEM = (Path(__file__).parent.parent.parent / "prompts" / "direct.txt").read_text()

def _get_structured_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.3)
    return llm.with_structured_output(AssistantResponse)


async def direct_node(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    user_text = messages[-1].content if messages else "Hello"

    try:
        result = await _get_structured_llm().ainvoke([
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": str(user_text)},
        ])
        response_dict = result.model_dump()
    except Exception:
        response_dict = AssistantResponse(
            message="Hi! I'm Billy, your accounting assistant. I can help with invoices, quotes, customers, products, emails, and more.",
            suggestions=["List invoices", "Create a new customer", "Show invoice summary"],
        ).model_dump()

    return {**state, "response": response_dict}
