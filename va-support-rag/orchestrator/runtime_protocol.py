"""AgentRuntime protocol — the shared contract all runtime backends must satisfy."""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from orchestrator.schemas import AgentInput, AgentOutput, StreamEvent


@runtime_checkable
class AgentRuntime(Protocol):
    async def run(self, input: AgentInput) -> AgentOutput: ...
    def stream(self, input: AgentInput) -> AsyncIterator[StreamEvent]: ...
    async def resume(self, thread_id: str, value: object) -> AgentOutput: ...
    def stream_resume(
        self, thread_id: str, value: object
    ) -> AsyncIterator[StreamEvent]: ...


__all__ = ["AgentRuntime"]
