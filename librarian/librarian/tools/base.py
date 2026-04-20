"""Base Pydantic models for tool input/output schemas."""

from __future__ import annotations

from pydantic import BaseModel


class ToolInput(BaseModel):
    """Override in subclasses to define tool input schema."""


class ToolOutput(BaseModel):
    """Override in subclasses to define tool output schema."""
