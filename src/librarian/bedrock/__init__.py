"""Backward-compatible re-export — canonical location is ``clients.bedrock``."""

from __future__ import annotations

from clients.bedrock import BedrockKBClient, BedrockKBResponse

__all__ = ["BedrockKBClient", "BedrockKBResponse"]
