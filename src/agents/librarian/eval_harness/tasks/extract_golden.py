"""Golden sample extraction — tiered quality levels.

Builds a golden evaluation dataset from three sources:

    gold    — hand-curated entries with verified chunk IDs.
              Highest trust: a human checked both the query and the exact
              chunks that answer it.

    silver  — human-validated entries where the expected_doc_url is confirmed
              correct but chunk IDs may be empty or approximate.

    bronze  — inferred from interaction logs (e.g. clicks, thumbs-up signals).
              Assumed correct but not verified. Useful for scale, lower trust.

    synthetic — LLM-generated (see generate_synthetic.py). Labelled explicitly
                so they can be filtered out of production eval runs.

Deduplication is keyed on (query, expected_doc_url) — NOT on record/ticket ID,
which may be absent or duplicated across export runs.

CLI::

    python -m agents.librarian.eval_harness.tasks.extract_golden \\
        --records data/records.jsonl \\
        --tier silver \\
        --output data/golden_silver.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from agents.librarian.eval_harness.tasks.models import GoldenSample
from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)

_VALID_TIERS = {"gold", "silver", "bronze", "synthetic"}


def load_records(path: str | Path) -> list[dict]:
    """Load raw records from a JSONL file.

    Each line must be a JSON object.  Lines that fail to parse are skipped
    with a warning (graceful degradation over strict failure).
    """
    records = []
    skipped = 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("extract_golden.parse_error", line=i, path=str(path))
                skipped += 1
    log.info("extract_golden.loaded", n=len(records), skipped=skipped, path=str(path))
    return records


def _make_query_id(query: str, doc_url: str) -> str:
    """Stable query ID from (query, doc_url) — deterministic across runs."""
    import hashlib

    key = f"{query.strip().lower()}|{doc_url.strip().lower()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def extract_samples(
    records: Iterable[dict],
    tier: str = "silver",
    *,
    query_field: str = "query",
    url_field: str = "expected_doc_url",
    chunk_ids_field: str = "relevant_chunk_ids",
    category_field: str = "category",
    language_field: str = "language",
    difficulty_field: str = "difficulty",
    validation_level_field: str = "validation_level",
    source_ticket_id_field: str = "source_ticket_id",
) -> list[GoldenSample]:
    """Extract GoldenSample objects from raw record dicts.

    Records that lack ``query_field`` or ``url_field`` are skipped.  Records
    whose ``validation_level`` (if present) does not match ``tier`` are also
    skipped, unless ``validation_level`` is absent — in that case the record
    is accepted and tagged with ``tier``.

    Args:
        records:   Iterable of raw record dicts (e.g. from load_records).
        tier:      Target validation level: gold | silver | bronze | synthetic.
        *_field:   Field name overrides for non-standard record schemas.

    Returns:
        Deduplicated list of GoldenSample objects.
    """
    if tier not in _VALID_TIERS:
        raise ValueError(f"Unknown tier {tier!r}. Valid: {sorted(_VALID_TIERS)}")

    seen: set[str] = set()
    samples: list[GoldenSample] = []
    skipped_missing = 0
    skipped_tier = 0
    skipped_dup = 0

    for rec in records:
        query = rec.get(query_field, "").strip()
        url = rec.get(url_field, "").strip()

        if not query or not url:
            skipped_missing += 1
            continue

        rec_tier = rec.get(validation_level_field, tier)
        if rec_tier != tier:
            skipped_tier += 1
            continue

        dedup_key = _make_query_id(query, url)
        if dedup_key in seen:
            skipped_dup += 1
            continue
        seen.add(dedup_key)

        samples.append(
            GoldenSample(
                query_id=dedup_key,
                query=query,
                expected_doc_url=url,
                relevant_chunk_ids=rec.get(chunk_ids_field) or [],
                category=rec.get(category_field, ""),
                language=rec.get(language_field, "en"),
                difficulty=rec.get(difficulty_field, "medium"),
                validation_level=tier,
                source_ticket_id=rec.get(source_ticket_id_field, ""),
            )
        )

    log.info(
        "extract_golden.extract.done",
        tier=tier,
        n_extracted=len(samples),
        skipped_missing=skipped_missing,
        skipped_tier=skipped_tier,
        skipped_dup=skipped_dup,
    )
    return samples


def filter_by_tier(
    samples: list[GoldenSample],
    tiers: list[str],
) -> list[GoldenSample]:
    """Return only samples whose validation_level is in ``tiers``."""
    tier_set = set(tiers)
    return [s for s in samples if s.validation_level in tier_set]


def save_samples(samples: list[GoldenSample], path: str | Path) -> None:
    """Write GoldenSample list to a JSONL file (one JSON object per line)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for sample in samples:
            f.write(sample.model_dump_json() + "\n")
    log.info("extract_golden.saved", n=len(samples), path=str(out))


def load_samples(path: str | Path) -> list[GoldenSample]:
    """Load GoldenSample list from a JSONL file produced by save_samples."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(GoldenSample.model_validate_json(line))
    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a golden evaluation dataset from a JSONL records file."
    )
    parser.add_argument(
        "--records",
        required=True,
        help="Path to input JSONL file (one record per line).",
    )
    parser.add_argument(
        "--tier",
        default="silver",
        choices=sorted(_VALID_TIERS),
        help="Validation level to extract (default: silver).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to <records_stem>_<tier>.jsonl beside the input.",
    )
    parser.add_argument(
        "--query-field", default="query", help="Record field for the query string."
    )
    parser.add_argument(
        "--url-field",
        default="expected_doc_url",
        help="Record field for the expected document URL.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    records_path = Path(args.records)
    if not records_path.exists():
        print(f"error: records file not found: {records_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or records_path.with_name(
        f"{records_path.stem}_{args.tier}.jsonl"
    )

    records = load_records(records_path)
    samples = extract_samples(
        records,
        tier=args.tier,
        query_field=args.query_field,
        url_field=args.url_field,
    )
    save_samples(samples, output_path)
    print(f"Extracted {len(samples)} {args.tier} samples → {output_path}")


if __name__ == "__main__":
    main()
