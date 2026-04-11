"""Synthetic golden sample generation.

Generates (query, expected_doc_url) pairs from a corpus of chunks using an
LLM (Haiku by default).  Output is JSONL-compatible with GoldenSample and
can be loaded directly by extract_golden.load_samples().

Cost gate: CONFIRM_EXPENSIVE_OPS must be True before any LLM calls are made.
Each chunk produces ~1 API call.  Estimated cost: ~$0.002–0.005 per sample
with Haiku.  Never commit CONFIRM_EXPENSIVE_OPS = True.

Typical usage (script)::

    CONFIRM_EXPENSIVE_OPS=true python -m \\
        agents.librarian.rag_core.eval_harness.tasks.generate_synthetic \\
        --chunks data/chunks.jsonl \\
        --output data/golden_synthetic.jsonl \\
        --n 100

The generated JSONL has validation_level="synthetic" so it can be filtered
out of production eval runs via extract_golden.filter_by_tier().
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import anthropic

from agents.librarian.rag_core.eval_harness.tasks.models import GoldenSample
from agents.librarian.rag_core.eval_harness.tasks.extract_golden import (
    _make_query_id,
    save_samples,
)
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)

CONFIRM_EXPENSIVE_OPS = False  # never commit as True

HAIKU_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM = """\
You are a question generation assistant.  Given a passage of text from a
documentation page, generate one realistic user question that the passage
directly answers.

Rules:
- The question must be answerable ONLY from the passage (no outside knowledge).
- Write it as a natural user query (not a test question or quiz item).
- Do not reference the passage explicitly (e.g. do NOT say "according to the text").
- Return ONLY a JSON object with this exact shape:
  {"query": "<the question>", "difficulty": "easy"|"medium"|"hard"}
No other text."""

_USER_TMPL = "Passage:\n{text}\n\nURL: {url}"


def _generate_one(
    client: anthropic.Anthropic,
    text: str,
    url: str,
    model: str = HAIKU_MODEL,
) -> dict[str, Any] | None:
    """Call the LLM for a single chunk.  Returns parsed dict or None on error."""
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            system=_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": _USER_TMPL.format(text=text[:2000], url=url),
                }
            ],
        )
        raw = resp.content[0].text.strip()
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("generate_synthetic.parse_error", error=str(exc))
        return None
    except anthropic.APIError as exc:
        log.error("generate_synthetic.api_error", error=str(exc))
        return None


def generate_from_chunks(
    chunks: list[dict],
    *,
    n: int | None = None,
    model: str = HAIKU_MODEL,
    text_field: str = "text",
    url_field: str = "url",
    chunk_id_field: str = "chunk_id",
) -> list[GoldenSample]:
    """Generate synthetic GoldenSample objects from chunk dicts.

    Args:
        chunks:       List of chunk dicts, each with at least ``text_field``
                      and ``url_field``.
        n:            Maximum number of samples to generate. Defaults to all chunks.
        model:        Anthropic model ID.
        text_field:   Key for the chunk text in each dict.
        url_field:    Key for the chunk's source URL.
        chunk_id_field: Key for the chunk ID (used as relevant_chunk_ids).

    Returns:
        List of GoldenSample with validation_level="synthetic".

    Raises:
        RuntimeError: If CONFIRM_EXPENSIVE_OPS is False.
    """
    if not CONFIRM_EXPENSIVE_OPS:
        raise RuntimeError(
            "Set CONFIRM_EXPENSIVE_OPS=True to run synthetic generation. "
            "Estimated cost: ~$0.002–0.005 per sample with Haiku."
        )

    client = anthropic.Anthropic()
    target = chunks[:n] if n is not None else chunks
    samples: list[GoldenSample] = []

    for i, chunk in enumerate(target):
        text = chunk.get(text_field, "")
        url = chunk.get(url_field, "")
        chunk_id = chunk.get(chunk_id_field, "")

        if not text or not url:
            log.warning("generate_synthetic.skip.missing_fields", index=i)
            continue

        result = _generate_one(client, text=text, url=url, model=model)
        if result is None:
            continue

        query = result.get("query", "").strip()
        difficulty = result.get("difficulty", "medium")
        if not query:
            log.warning("generate_synthetic.skip.empty_query", index=i)
            continue

        query_id = _make_query_id(query, url)
        samples.append(
            GoldenSample(
                query_id=query_id,
                query=query,
                expected_doc_url=url,
                relevant_chunk_ids=[chunk_id] if chunk_id else [],
                category="",
                language="en",
                difficulty=difficulty,
                validation_level="synthetic",
                source_ticket_id="",
            )
        )
        log.debug("generate_synthetic.sample.created", query_id=query_id, url=url)

    log.info(
        "generate_synthetic.done",
        n_chunks=len(target),
        n_generated=len(samples),
    )
    return samples


def load_chunks_jsonl(path: str | Path) -> list[dict]:
    """Load chunk dicts from a JSONL file (one JSON object per line)."""
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic golden samples from a JSONL chunks file."
    )
    parser.add_argument(
        "--chunks",
        required=True,
        help="Path to input JSONL file (one chunk dict per line).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to <chunks_stem>_synthetic.jsonl.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Maximum number of samples to generate (default: all chunks).",
    )
    parser.add_argument(
        "--model",
        default=HAIKU_MODEL,
        help=f"Anthropic model ID (default: {HAIKU_MODEL}).",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Chunk field for the passage text (default: text).",
    )
    parser.add_argument(
        "--url-field",
        default="url",
        help="Chunk field for the source URL (default: url).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"error: chunks file not found: {chunks_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or chunks_path.with_name(
        f"{chunks_path.stem}_synthetic.jsonl"
    )

    # Require the env-level gate when invoked from CLI
    import os

    global CONFIRM_EXPENSIVE_OPS
    if os.getenv("CONFIRM_EXPENSIVE_OPS", "").lower() in ("1", "true", "yes"):
        CONFIRM_EXPENSIVE_OPS = True

    chunks = load_chunks_jsonl(chunks_path)
    samples = generate_from_chunks(
        chunks,
        n=args.n,
        model=args.model,
        text_field=args.text_field,
        url_field=args.url_field,
    )
    save_samples(samples, output_path)
    print(f"Generated {len(samples)} synthetic samples → {output_path}")


if __name__ == "__main__":
    main()
