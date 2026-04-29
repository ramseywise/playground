"""Base classes for framework-agnostic tools."""

from __future__ import annotations

from pydantic import BaseModel


class ToolInput(BaseModel):
    """Base input schema for all tools."""


class ToolOutput(BaseModel):
    """Base output schema for all tools."""


__all__ = ["ToolInput", "ToolOutput"]
