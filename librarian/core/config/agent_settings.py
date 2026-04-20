"""Agent-specific configuration — extends infra.config.BaseSettings.

Researcher and presenter settings live here; librarian has its own
``LibrarySettings`` that also extends ``BaseSettings``.
"""

from __future__ import annotations

from pathlib import Path

from core.config.settings import BaseSettings


class Settings(BaseSettings):
    """Settings for researcher + presenter agents.  Extends shared BaseSettings."""

    @property
    def anthropic_model(self) -> str:
        return self.model_sonnet

    # Research agent paths
    readings_dir: Path = Path.home() / "Dropbox" / "ai_readings"
    obsidian_vault: Path = Path.home() / "workspace" / "obsidian"
    pdftotext_bin: Path = Path("/opt/homebrew/bin/pdftotext")
    pdfinfo_bin: Path = Path("/opt/homebrew/bin/pdfinfo")

    # Visualizer settings
    image_provider: str = "pollinations"  # "pollinations" | "replicate"
    pollinations_model: str = "flux"
    pollinations_seed: int | None = None  # None = random
    pollinations_enhance: bool = False
    replicate_api_token: str = ""
    viz_output_dir: Path = Path("output")
    image_width: int = 1280
    image_height: int = 720
    viz_audience: str = "mixed technical and product team"
    viz_model: str = "claude-sonnet-4-6"

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
