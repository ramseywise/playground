"""Backward-compatible re-export — canonical location is ``clients.bedrock``."""

from __future__ import annotations

from clients.bedrock_KB import BedrockKBClient, BedrockKBResponse

__all__ = ["BedrockKBClient", "BedrockKBResponse"]
