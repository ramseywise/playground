"""Loaders that convert external eval datasets into GoldenSample lists.

Data files live OUTSIDE the project (env-var controlled path) and are
never committed to this repo.  Each loader is a pure function:
    path -> list[GoldenSample]

Supported formats:
  load_golden_from_jsonl  — cs_agent_assist_with_rag eval_dataset.jsonl schema
  load_golden_from_faq_csv — scraped FAQ CSV (url + question/query + answer/text columns)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from evals.utils.models import EvalTask, GoldenSample


def load_golden_from_jsonl(path: str | Path) -> list[GoldenSample]:
    """Load GoldenSamples from cs_agent_assist_with_rag eval_dataset.jsonl.

    Field mapping:
      query_id          -> query_id
      query             -> query
      source_doc_id     -> expected_doc_url  (doc-level ground truth)
      relevant_chunks   -> relevant_chunk_ids
      category          -> category
      retrieval_scores  -> difficulty (derived: see _difficulty_from_scores)
      (all queries are German)

    Args:
        path: Absolute path to the .jsonl file.

    Returns:
        List of GoldenSample objects ready for evaluate_retrieval().

    Raises:
        FileNotFoundError: If the file does not exist at path.
        ValueError: If the file contains no valid records.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"eval dataset not found: {path}")

    samples: list[GoldenSample] = []
    seen_ids: set[str] = set()

    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid JSON — {exc}") from exc

            query_id = record.get("query_id", f"row_{line_no}")

            # Deduplicate — the dataset contains repeated queries
            if query_id in seen_ids:
                continue
            seen_ids.add(query_id)

            samples.append(
                GoldenSample(
                    query_id=query_id,
                    query=record["query"],
                    expected_doc_url=record.get("source_doc_id", ""),
                    relevant_chunk_ids=record.get("relevant_chunks", []),
                    category=record.get("category", ""),
                    language="de",
                    difficulty=_difficulty_from_scores(
                        record.get("retrieval_scores", [])
                    ),
                    validation_level="silver",
                )
            )

    if not samples:
        raise ValueError(f"no valid records in {path}")

    return samples


_CHUNK_ID_SPLIT_RE = re.compile(r"[\|,;]+")


def _parse_relevant_chunk_ids(raw: str) -> list[str]:
    """Parse ``relevant_chunks`` cell: JSON array, pipe/csv list, or single id."""
    s = (raw or "").strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        return []
    return [p.strip() for p in _CHUNK_ID_SPLIT_RE.split(s) if p.strip()]


def load_golden_from_faq_csv(
    path: str | Path,
    *,
    limit: int | None = None,
) -> list[GoldenSample]:
    """Load :class:`GoldenSample` rows from a scraped FAQ CSV.

    Uses the same column conventions as :mod:`app.rag.preprocessing.faq_csv`
    (case-insensitive headers). Each eval row needs a **query** (``query`` /
    ``question`` / …) and **expected URL** (``url`` / ``link`` / ``source_doc_id`` / …).

    Optional columns: ``query_id``, ``category``, ``language``, ``relevant_chunks``
    (JSON array or ``id1|id2``), ``difficulty``, ``validation_level``.

    Rows without both query and URL are skipped.

    Args:
        path: CSV file.
        limit: If set, stop after this many accepted rows (e.g. ``50`` for a smoke slice).
    """
    from rag.preprocessing.faq_csv import (
        eval_query_and_doc_url,
        iter_normalized_faq_rows,
    )

    path = Path(path)
    samples: list[GoldenSample] = []
    seen_ids: set[str] = set()

    for line_no, n in iter_normalized_faq_rows(path):
        q, url = eval_query_and_doc_url(n)
        if not q or not url:
            continue

        qid = (n.get("query_id") or "").strip() or f"faq_line_{line_no}"
        if qid in seen_ids:
            continue
        seen_ids.add(qid)

        cat = (n.get("category") or n.get("topic") or "").strip()
        lang = (n.get("language") or n.get("lang") or "en").strip() or "en"
        diff = (n.get("difficulty") or "medium").strip() or "medium"
        val = (n.get("validation_level") or "silver").strip() or "silver"
        chunks = _parse_relevant_chunk_ids(n.get("relevant_chunks", ""))

        samples.append(
            GoldenSample(
                query_id=qid,
                query=q,
                expected_doc_url=url,
                relevant_chunk_ids=chunks,
                category=cat,
                language=lang,
                difficulty=diff,
                validation_level=val,
            )
        )
        if limit is not None and len(samples) >= limit:
            break

    if not samples:
        raise ValueError(
            f"no eval rows with both query and url in {path} — check column names"
        )
    return samples


def golden_samples_to_eval_tasks(samples: list[GoldenSample]) -> list[EvalTask]:
    """Map golden retrieval labels to :class:`~evals.utils.models.EvalTask` for regression harness."""

    return [
        EvalTask(
            id=s.query_id,
            query=s.query,
            expected_answer=s.expected_doc_url or "",
            metadata={
                "expected_doc_url": s.expected_doc_url,
                "relevant_chunk_ids": s.relevant_chunk_ids,
            },
            category=s.category,
            difficulty=s.difficulty,
            tags=["faq"],
        )
        for s in samples
    ]


def _difficulty_from_scores(scores: list[float]) -> str:
    """Derive difficulty from the best retrieval score in the original dataset.

    High scores mean the corpus already retrieved well → semantically easy.
    Low scores suggest the query is hard to match.
    """
    if not scores:
        return "medium"
    best = max(scores)
    if best >= 0.60:
        return "easy"
    if best >= 0.40:
        return "medium"
    return "hard"
