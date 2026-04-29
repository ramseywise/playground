"""Load scraped FAQ CSV rows into ingestion ``doc`` dicts.

Expected shapes (headers are matched case-insensitively):

- **Canonical FAQ:** ``url`` (or ``link``, ``source_doc_id``, …) + ``question`` + ``answer``
  (or ``query`` / ``body`` / ``content``). Indexed text defaults to
  ``Question: …\\n\\nAnswer: …`` so retrieval matches natural user questions.

- **Single body:** ``url`` + ``text`` (optional ``query`` / ``question`` for eval-only
  datasets that also ship content in ``text``).

Rows without a URL or without ingestible text are skipped with a debug log.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterator

from rag.preprocessing.chunking.utils import stable_doc_id_from_document

log = logging.getLogger(__name__)

_URL_KEYS = (
    "url",
    "link",
    "href",
    "page_url",
    "canonical_url",
    "source_doc_id",
)
_QUERY_KEYS = ("query", "question", "q", "user_query")
_TITLE_KEYS = ("title", "heading", "name")
_TOPIC_KEYS = ("topic", "category", "section", "tag")


def _normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _row_dict(raw: dict[str, str | None]) -> dict[str, str]:
    return {_normalize_header(k): (v or "").strip() for k, v in raw.items() if k}


def _first(n: dict[str, str], *keys: str) -> str:
    for key in keys:
        v = n.get(key, "").strip()
        if v:
            return v
    return ""


def eval_query_and_doc_url(n: dict[str, str]) -> tuple[str, str]:
    """Golden eval: user ``query`` and ``expected_doc_url`` from a normalized CSV row."""
    return _first(n, *_QUERY_KEYS), _first(n, *_URL_KEYS)


def _doc_text(n: dict[str, str]) -> tuple[str, str]:
    """Return (full_text_for_index, short_label_for_title_fallback)."""
    q = _first(n, *_QUERY_KEYS)
    # Prefer explicit answer columns; else use text/full_text as the body.
    a = _first(n, "answer", "body", "content")
    if not a:
        a = _first(n, "text", "full_text")

    if q and a:
        return (f"Question: {q}\n\nAnswer: {a}", q)
    if a and not q:
        return (a, _first(n, *_TITLE_KEYS) or "faq")
    return ("", "")


def iter_normalized_faq_rows(
    path: str | Path,
) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield ``(line_no, normalized_row)`` for each CSV data row (UTF-8).

    *line_no* is the 1-based file line number (useful for stable ``query_id``).
    Keys in *normalized_row* are lowercased, underscores, stripped.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FAQ CSV not found: {path}")

    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {path}")

        for i, raw in enumerate(reader, start=2):
            yield i, _row_dict(raw)


def iter_faq_csv_docs(path: str | Path) -> Iterator[dict[str, str]]:
    """Yield ingestion document dicts from *path* (UTF-8 CSV)."""
    path = Path(path)
    for i, n in iter_normalized_faq_rows(path):
        url = _first(n, *_URL_KEYS)
        text, title_hint = _doc_text(n)
        if not url:
            log.debug("faq_csv.skip row=%d reason=no_url", i)
            continue
        if not text:
            log.debug("faq_csv.skip row=%d reason=no_text", i)
            continue

        title = _first(n, *_TITLE_KEYS) or title_hint
        topic = _first(n, *_TOPIC_KEYS)
        source = _first(n, "source") or "scraped_faq"
        source_file = f"{path.name}:line_{i}"

        doc = {
            "text": text,
            "title": title,
            "url": url,
            "source": source,
            "content_type": "faq",
            "topic": topic,
            "source_file": source_file,
            "section": "",
        }
        doc["stable_doc_id"] = stable_doc_id_from_document(doc)
        yield doc


def load_faq_csv_documents(path: str | Path) -> list[dict[str, str]]:
    """Load all FAQ rows as document dicts for :class:`~app.rag.preprocessing.ingestion.IngestionPipeline`."""
    return list(iter_faq_csv_docs(path))
