"""Shared configuration for all agents — loaded from .env via pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central config for all agents. Every field is overridable via .env or env vars."""

    anthropic_api_key: str = ""  # validated at call time, not import time
    anthropic_model: str = "claude-sonnet-4-6"

    # Research agent paths
    readings_dir: Path = Path.home() / "Dropbox" / "ai_readings"
    obsidian_vault: Path = Path.home() / "workspace" / "obsidian"
    pdftotext_bin: Path = Path("/opt/homebrew/bin/pdftotext")
    pdfinfo_bin: Path = Path("/opt/homebrew/bin/pdfinfo")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def project_context_file(self) -> Path:
        return self.obsidian_vault / "project_context.md"


def load_project_context() -> str:
    """Load the project context brief for research note synthesis.

    Returns empty string if the file doesn't exist — notes will omit
    the 'Relevance to Active Work' connections rather than failing.
    """
    path = settings.project_context_file
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


settings = Settings()
