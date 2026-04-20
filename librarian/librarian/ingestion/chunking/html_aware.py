from __future__ import annotations

import re
import uuid

from librarian.ingestion.base import ChunkerConfig
from librarian.ingestion.chunking.utils import (
    make_doc_id,
    recursive_split_by_separators,
    word_count,
)
from librarian.schemas.chunks import Chunk, ChunkMetadata

# Backward-compat aliases
_make_doc_id = make_doc_id
_word_count = word_count
_recursive_split = recursive_split_by_separators

# Matches heading markers left by html-to-text converters, e.g. "## Heading"
_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)


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
