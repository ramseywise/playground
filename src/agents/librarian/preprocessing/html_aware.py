from __future__ import annotations

import hashlib
import re
import uuid

from agents.librarian.preprocessing.base import ChunkerConfig
from agents.librarian.schemas.chunks import Chunk, ChunkMetadata

# Matches heading markers left by html-to-text converters, e.g. "## Heading"
_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)


def _make_doc_id(url: str, section: str | None) -> str:
    raw = f"{url}::{section or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _word_count(text: str) -> int:
    return len(text.split())


def _recursive_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text using progressively finer separators until chunks fit max_tokens."""
    if _word_count(text) <= max_tokens:
        return [text]

    for sep in ("\n\n", "\n", ". ", " "):
        parts = text.split(sep)
        if len(parts) > 1:
            chunks: list[str] = []
            current = ""
            for part in parts:
                candidate = (current + sep + part).strip() if current else part.strip()
                if _word_count(candidate) <= max_tokens:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # carry overlap from tail of previous chunk
                    overlap_words = current.split()[-overlap_tokens:] if current else []
                    current = (" ".join(overlap_words) + " " + part).strip()
            if current:
                chunks.append(current)
            # recurse only if we made progress
            if len(chunks) > 1:
                result: list[str] = []
                for c in chunks:
                    result.extend(_recursive_split(c, max_tokens, overlap_tokens))
                return result

    # fallback: word-level hard split
    words = text.split()
    return [
        " ".join(words[i : i + max_tokens])
        for i in range(0, len(words), max_tokens - overlap_tokens)
        if words[i : i + max_tokens]
    ]


class HtmlAwareChunker:
    """Splits plain-text documents at heading boundaries, then recursively by size.

    Handles both ``doc["text"]`` and ``doc["full_text"]`` field names to accommodate
    variance across source connectors.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk_document(self, doc: dict) -> list[Chunk]:
        text: str = doc.get("text") or doc.get("full_text") or ""
        url: str = doc.get("url", "")
        title: str = doc.get("title", "")
        language: str = doc.get("language", "en")
        extra_meta: dict = {
            k: v
            for k, v in doc.items()
            if k not in {"text", "full_text", "url", "title", "language"}
        }

        sections = self._split_sections(text)
        chunks: list[Chunk] = []

        for section_title, body in sections:
            for fragment in _recursive_split(
                body.strip(), self.config.max_tokens, self.config.overlap_tokens
            ):
                if _word_count(fragment) < self.config.min_tokens:
                    continue
                doc_id = _make_doc_id(url, section_title)
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        text=fragment,
                        metadata=ChunkMetadata(
                            url=url,
                            title=title,
                            doc_id=doc_id,
                            section=section_title,
                            language=language,
                            namespace=extra_meta.get("namespace"),
                            topic=extra_meta.get("topic"),
                            content_type=extra_meta.get("content_type"),
                            access_tier=extra_meta.get("access_tier"),
                            source_id=extra_meta.get("source_id"),
                        ),
                    )
                )

        return chunks

    def _split_sections(self, text: str) -> list[tuple[str | None, str]]:
        """Return list of (section_title, body) pairs split at h1–h3 headings."""
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return [(None, text)]

        sections: list[tuple[str | None, str]] = []
        # text before first heading
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

        for i, match in enumerate(matches):
            section_title = match.group(1).strip()
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            sections.append((section_title, body))

        return sections
