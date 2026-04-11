"""Text cleaning and boilerplate removal."""

from __future__ import annotations

import re

from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)

_NOISE_PATTERNS: list[str] = [
    r"Sent from my \w+",
    r"Get Outlook for .*",
    r"--\s*\n.*$",  # email signature separator
]


def clean_text(text: str, extra_noise_patterns: list[str] | None = None) -> str:
    """Normalize whitespace and strip common noise (email signatures, app footers).

    Args:
        text: Raw input text.
        extra_noise_patterns: Additional regex patterns to strip (e.g. brand-specific boilerplate).

    Returns:
        Cleaned text with normalized whitespace.
    """
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    for pattern in _NOISE_PATTERNS + (extra_noise_patterns or []):
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    return text.strip()


def remove_boilerplate(text: str, patterns: list[str] | None = None) -> str:
    """Remove boilerplate content from documents.

    Args:
        text: Input text.
        patterns: Regex patterns to strip. Applied in addition to nothing (no defaults —
            pass domain-specific patterns explicitly to keep this function generic).

    Returns:
        Text with matched patterns removed.
    """
    if not text or not patterns:
        return text or ""

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    return text.strip()
