"""Loaders that convert external eval datasets into GoldenSample lists.

Data files live OUTSIDE the project (env-var controlled path) and are
never committed to this repo.  Each loader is a pure function:
    path -> list[GoldenSample]

Supported formats:
  load_golden_from_jsonl  — cs_agent_assist_with_rag eval_dataset.jsonl schema
"""

from __future__ import annotations

import json
from pathlib import Path

from librarian.ingestion.tasks.models import GoldenSample


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
