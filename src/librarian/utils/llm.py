"""Re-export from canonical location: infra.clients.llm.

The librarian's LLM wrapper now lives in the shared infra layer.
This module re-exports for backward compatibility within the librarian.
"""

from __future__ import annotations

from agents.librarian.tools.core.clients.llm import AnthropicLLM  # noqa: F401

__all__ = ["AnthropicLLM"]
