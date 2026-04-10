"""Golden dataset loading and saving utilities.

Common patterns for loading/saving evaluation datasets as JSON/Parquet.
Agent-specific datasets call these helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_golden_json(path: Path) -> list[dict[str, Any]]:
    """Load golden samples from a JSON file.

    Expects a JSON array of objects, each representing one evaluation sample.
    """
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        msg = f"Expected JSON array, got {type(data).__name__}"
        raise ValueError(msg)
    return data


def save_golden_json(samples: list[dict[str, Any]], path: Path) -> None:
    """Save golden samples to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False, default=str)
