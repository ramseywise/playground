"""Token counting (tiktoken ``cl100k_base``) — shared by ingestion, evals, and CLI reports.

Embedding models use :mod:`app.rag.embedding` (SentenceTransformers). Token counts here
are for **budgeting / logging** against OpenAI-style limits, not for the embedder.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import tiktoken

log = logging.getLogger(__name__)

_encoding: tiktoken.Encoding | None = None


def _encoding_cl100k() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens(text: str) -> int:
    """Count tokens using ``cl100k_base`` (GPT-4 class)."""
    if not text:
        return 0
    return len(_encoding_cl100k().encode(text))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_document_dir() -> Path:
    env = os.getenv("RAG_DATA_DIR")
    base = Path(env).expanduser() if env else _repo_root() / "data"
    return base / "document"


def write_token_report(
    document_dir: Path | None = None,
    report_path: Path | None = None,
) -> Path:
    """Scan ``*.txt`` under *document_dir*, write a token summary next to it."""
    doc_dir = document_dir or _default_document_dir()
    out = report_path or (doc_dir / "token_report.txt")

    if not doc_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {doc_dir}")

    txt_files = sorted(doc_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {doc_dir}")

    lines: list[str] = ["Token Count Report", "=" * 50, ""]
    total = 0
    for fp in txt_files:
        content = fp.read_text(encoding="utf-8")
        n = count_tokens(content)
        total += n
        lines.append(f"File: {fp.name}")
        lines.append(f"Token count: {n}")
        lines.append("-" * 50)

    lines.extend(
        [
            "",
            "Summary",
            "=" * 50,
            f"Number of documents: {len(txt_files)}",
            f"Total tokens: {total}",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info(
        "tokenization.report path=%s docs=%d total_tokens=%d",
        out,
        len(txt_files),
        total,
    )
    return out


def main() -> None:
    """CLI: ``python -m app.rag.tokenization`` — writes ``token_report.txt`` under the document dir."""
    path = write_token_report()
    print(f"Token report saved to: {path}")  # noqa: T201


if __name__ == "__main__":
    main()
