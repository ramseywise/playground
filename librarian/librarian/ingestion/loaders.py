from __future__ import annotations

import re
from pathlib import Path

from core.logging import get_logger

log = get_logger(__name__)

# Matches a YAML-style frontmatter block at the top of a Markdown file.
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# Matches a single frontmatter key: value line (unquoted or double-quoted value).
_KV_RE = re.compile(r'^(\w[\w_-]*)\s*:\s*"?([^"\n]*)"?\s*$', re.MULTILINE)


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Extract YAML-lite frontmatter and return (metadata_dict, body_text)."""
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw

    fm_block = match.group(1)
    body = raw[match.end():]
    meta: dict[str, str] = {}
    for key, value in _KV_RE.findall(fm_block):
        cleaned = value.strip()
        if cleaned.lower() == "null":
            cleaned = ""
        meta[key] = cleaned
    return meta, body


def load_markdown_file(path: Path) -> dict[str, str]:
    """Load a single Markdown file and return a doc dict.

    Returns keys: text, title, url, source, content_type, topic, source_file.
    Frontmatter values override defaults; missing keys fall back to empty strings.
    """
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)

    doc = {
        "text": body.strip(),
        "title": meta.get("title", path.stem.replace("_", " ").title()),
        "url": meta.get("url", ""),
        "source": meta.get("source", "blog"),
        "content_type": meta.get("content_type", "article"),
        "topic": meta.get("topic", ""),
        "source_file": str(path),
    }
    log.debug("loader.markdown.loaded", path=str(path), title=doc["title"])
    return doc


def load_directory(
    directory: Path,
    glob_pattern: str = "*.md",
) -> list[dict[str, str]]:
    """Load all matching files from *directory* and return a sorted list of docs."""
    paths = sorted(directory.glob(glob_pattern))
    docs = [load_markdown_file(p) for p in paths]
    log.info("loader.directory.loaded", directory=str(directory), count=len(docs))
    return docs
