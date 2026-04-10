"""Generic task extraction from JSONL records.

Generalises the librarian's ``extract_golden.py`` pattern: load records,
filter by tier, deduplicate, and produce ``EvalTask`` objects.

Conversion helpers bridge between ``EvalTask`` and agent-specific models
(e.g. librarian's ``GoldenSample``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, cast

import structlog

from eval.models import EvalTask

log = structlog.get_logger(__name__)

VALID_TIERS = frozenset({"gold", "silver", "bronze", "synthetic"})


class _SupportsModelDump(Protocol):
    def model_dump(self) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load_records_jsonl(path: str | Path) -> list[dict]:
    """Load raw records from a JSONL file, skipping malformed lines."""
    records: list[dict] = []
    skipped = 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("extract.parse_error", line=i, path=str(path))
                skipped += 1
    log.info("extract.loaded", n=len(records), skipped=skipped)
    return records


def save_tasks_jsonl(tasks: list[EvalTask], path: str | Path) -> None:
    """Write EvalTask list to a JSONL file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for task in tasks:
            f.write(task.model_dump_json() + "\n")
    log.info("extract.saved", n=len(tasks), path=str(out))


def load_tasks_jsonl(path: str | Path) -> list[EvalTask]:
    """Load EvalTask list from a JSONL file."""
    tasks: list[EvalTask] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(EvalTask.model_validate_json(line))
    return tasks


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def make_task_id(query: str, answer: str) -> str:
    """Stable task ID from (query, answer) — deterministic across runs."""
    key = f"{query.strip().lower()}|{answer.strip().lower()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def extract_tasks(
    records: Iterable[dict],
    tier: str = "silver",
    *,
    query_field: str = "query",
    answer_field: str = "expected_answer",
    context_field: str = "context",
    category_field: str = "category",
    difficulty_field: str = "difficulty",
    validation_level_field: str = "validation_level",
) -> list[EvalTask]:
    """Extract EvalTask objects from raw record dicts.

    Records missing ``query_field`` are skipped.  Records whose
    ``validation_level`` doesn't match ``tier`` are also skipped.
    Deduplication is keyed on (query, expected_answer).
    """
    if tier not in VALID_TIERS:
        msg = f"Unknown tier {tier!r}. Valid: {sorted(VALID_TIERS)}"
        raise ValueError(msg)

    seen: set[str] = set()
    tasks: list[EvalTask] = []
    skipped_missing = 0
    skipped_tier = 0
    skipped_dup = 0

    for rec in records:
        query = rec.get(query_field, "").strip()
        if not query:
            skipped_missing += 1
            continue

        rec_tier = rec.get(validation_level_field, tier)
        if rec_tier != tier:
            skipped_tier += 1
            continue

        answer = rec.get(answer_field, "").strip()
        dedup_key = make_task_id(query, answer)
        if dedup_key in seen:
            skipped_dup += 1
            continue
        seen.add(dedup_key)

        tasks.append(
            EvalTask(
                id=dedup_key,
                query=query,
                expected_answer=answer,
                context=rec.get(context_field, ""),
                category=rec.get(category_field, ""),
                difficulty=rec.get(difficulty_field, "medium"),
                validation_level=tier,
            )
        )

    log.info(
        "extract.done",
        tier=tier,
        n_extracted=len(tasks),
        skipped_missing=skipped_missing,
        skipped_tier=skipped_tier,
        skipped_dup=skipped_dup,
    )
    return tasks


def filter_by_tier(tasks: list[EvalTask], tiers: list[str]) -> list[EvalTask]:
    """Return only tasks whose validation_level is in ``tiers``."""
    tier_set = set(tiers)
    return [t for t in tasks if t.validation_level in tier_set]


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def golden_sample_to_task(sample: object) -> EvalTask:
    """Convert a librarian GoldenSample (or any Pydantic model with matching fields) to EvalTask."""
    if hasattr(sample, "model_dump"):
        data = cast(_SupportsModelDump, sample).model_dump()
    else:
        data = vars(sample)
    return EvalTask(
        id=data.get("query_id", ""),
        query=data.get("query", ""),
        expected_answer=data.get("expected_doc_url", ""),
        category=data.get("category", ""),
        difficulty=data.get("difficulty", "medium"),
        validation_level=data.get("validation_level", "silver"),
        metadata={
            "expected_doc_url": data.get("expected_doc_url", ""),
            "relevant_chunk_ids": data.get("relevant_chunk_ids", []),
            "source_record_id": data.get(
                "source_record_id", data.get("source_ticket_id", "")
            ),
            "language": data.get("language", "en"),
        },
    )
