"""Base tool protocol — framework-agnostic contract for LangGraph and ADK."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class ToolInput(BaseModel):
    """Override in subclasses to define tool input schema."""


class ToolOutput(BaseModel):
    """Override in subclasses to define tool output schema."""


@runtime_checkable
class BaseTool(Protocol):
    """Minimal protocol that both LangGraph and ADK adapters can consume.

    Concrete tools declare ``name``, ``description``, typed schemas,
    and an async ``run()`` method.
    """

    name: str
    description: str
    input_schema: type[ToolInput]
    output_schema: type[ToolOutput]

    async def run(self, tool_input: ToolInput) -> ToolOutput: ...
