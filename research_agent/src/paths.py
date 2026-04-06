from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # src/ → research_agent/
DROPBOX_READINGS = Path.home() / "Dropbox" / "ai_readings"
OBSIDIAN_VAULT = Path.home() / "workspace" / "obsidian"
OBSIDIAN_TOPICS = OBSIDIAN_VAULT / "topics"
OBSIDIAN_INDEX = OBSIDIAN_VAULT / "_index.md"
PDFTOTEXT_BIN = Path("/opt/homebrew/bin/pdftotext")
PDFINFO_BIN = Path("/opt/homebrew/bin/pdfinfo")
