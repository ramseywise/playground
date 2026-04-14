"""Base tool protocol — framework-agnostic contract for LangGraph and ADK."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel


class ToolInput(BaseModel):
    """Override in subclasses to define tool input schema."""


class ToolOutput(BaseModel):
    """Override in subclasses to define tool output schema."""


InputT = TypeVar("InputT", bound=ToolInput)
OutputT = TypeVar("OutputT", bound=ToolOutput)


@runtime_checkable
class BaseTool(Protocol[InputT, OutputT]):
    """Minimal generic protocol that both LangGraph and ADK adapters can consume.

    Concrete tools declare ``name``, ``description``, typed schemas,
    and an async ``run()`` method.  Parameterize with concrete
    input/output types: ``BaseTool[RetrieverToolInput, RetrieverToolOutput]``.
    """

    name: str
    description: str
    input_schema: type[InputT]
    output_schema: type[OutputT]

    async def run(self, tool_input: InputT) -> OutputT: ...
