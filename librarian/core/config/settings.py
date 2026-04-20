"""Base settings shared by all agents.

Agent-specific settings extend ``BaseSettings`` and add their own fields.
API keys, model names, and log config are defined once here.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings as _PydanticBaseSettings, SettingsConfigDict


class BaseSettings(_PydanticBaseSettings):
    """Cross-agent configuration.  Every field is overridable via .env or env vars."""

    # LLM
    llm_provider: str = "anthropic"  # anthropic | gemini
    anthropic_api_key: str = ""  # validated at call time, not import time
    model_sonnet: str = "claude-sonnet-4-6"
    model_haiku: str = "claude-haiku-4-5-20251001"

    # Google Gemini
    gemini_api_key: str = ""
    model_gemini: str = "gemini-2.0-flash"

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
