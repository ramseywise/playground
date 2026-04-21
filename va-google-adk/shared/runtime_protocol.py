"""AgentRuntime Protocol — formal interface shared by LangGraph and ADK gateways.

Both GraphRunner (LangGraph) and SessionManager (ADK) implement this contract
so a single FastAPI app can swap backends via VA_BACKEND env var.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from pydantic import BaseModel


class AgentInput(BaseModel):
    session_id: str
    message: str
    page_url: str | None = None
    trace_id: str | None = None


class AgentOutput(BaseModel):
    response: dict


class StreamEvent(BaseModel):
    type: str
    data: dict | str | None = None
    trace_id: str | None = None


@runtime_checkable
class AgentRuntime(Protocol):
    async def run(self, input: AgentInput) -> AgentOutput: ...
    def stream(self, input: AgentInput) -> AsyncIterator[StreamEvent]: ...
    async def resume(self, thread_id: str, value: object) -> AgentOutput: ...
