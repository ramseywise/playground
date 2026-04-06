"""Tests for batch processing and manifest management in __main__.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.research.__main__ import (
    MANIFEST_PATH,
    _find_unprocessed,
    _load_manifest,
    _pdf_hash,
    _relative_key,
    _save_manifest,
)


@pytest.fixture()
def readings_dir(tmp_path: Path) -> Path:
    """Create a fake readings directory with sample PDFs."""
    readings = tmp_path / "ai_readings"
    (readings / "0.rag").mkdir(parents=True)
    (readings / "2.knowledge graphs").mkdir(parents=True)

    # Create fake PDFs (just need to be files with .pdf extension)
    (readings / "0.rag" / "paper1.pdf").write_bytes(b"fake pdf content 1")
    (readings / "0.rag" / "paper2.pdf").write_bytes(b"fake pdf content 2")
    (readings / "2.knowledge graphs" / "ch13.pdf").write_bytes(b"fake pdf content 3")

    return readings


@pytest.fixture()
def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "vault" / ".processed.json"


# --- Manifest I/O ---


def test_load_manifest_returns_empty_when_missing(tmp_path: Path) -> None:
    with patch("agents.research.__main__.MANIFEST_PATH", tmp_path / "nonexistent.json"):
        manifest = _load_manifest()
    assert manifest == {}


def test_save_and_load_manifest(tmp_path: Path) -> None:
    manifest_file = tmp_path / ".processed.json"
    with patch("agents.research.__main__.MANIFEST_PATH", manifest_file):
        data = {"0.rag/paper1.pdf": {"date": "2026-04-06", "hash": "abc123", "note": "paper1"}}
        _save_manifest(data)
        loaded = _load_manifest()

    assert loaded == data


def test_save_manifest_creates_parent_dirs(tmp_path: Path) -> None:
    manifest_file = tmp_path / "nested" / "dir" / ".processed.json"
    with patch("agents.research.__main__.MANIFEST_PATH", manifest_file):
        _save_manifest({"key": {"val": "data"}})
    assert manifest_file.exists()


# --- PDF hashing ---


def test_pdf_hash_consistent(tmp_path: Path) -> None:
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"some content")
    h1 = _pdf_hash(pdf)
    h2 = _pdf_hash(pdf)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest


def test_pdf_hash_differs_for_different_content(tmp_path: Path) -> None:
    pdf1 = tmp_path / "a.pdf"
    pdf2 = tmp_path / "b.pdf"
    pdf1.write_bytes(b"content A")
    pdf2.write_bytes(b"content B")
    assert _pdf_hash(pdf1) != _pdf_hash(pdf2)


# --- Finding unprocessed PDFs ---


def test_find_unprocessed_all_new(readings_dir: Path) -> None:
    with patch("agents.research.__main__.settings") as mock_settings:
        mock_settings.readings_dir = readings_dir
        unprocessed = _find_unprocessed({})

    assert len(unprocessed) == 3


def test_find_unprocessed_skips_already_processed(readings_dir: Path) -> None:
    pdf_path = readings_dir / "0.rag" / "paper1.pdf"

    with patch("agents.research.__main__.settings") as mock_settings:
        mock_settings.readings_dir = readings_dir

        key = _relative_key(pdf_path)
        manifest = {key: {"date": "2026-04-06", "hash": _pdf_hash(pdf_path), "note": "paper1"}}
        unprocessed = _find_unprocessed(manifest)

    assert len(unprocessed) == 2
    assert pdf_path not in unprocessed


def test_find_unprocessed_detects_changed_hash(readings_dir: Path) -> None:
    pdf_path = readings_dir / "0.rag" / "paper1.pdf"

    with patch("agents.research.__main__.settings") as mock_settings:
        mock_settings.readings_dir = readings_dir

        key = _relative_key(pdf_path)
        manifest = {key: {"date": "2026-04-06", "hash": "stale_hash", "note": "paper1"}}
        unprocessed = _find_unprocessed(manifest)

    # paper1.pdf should be re-processed because hash changed
    assert len(unprocessed) == 3
    assert pdf_path in unprocessed


def test_find_unprocessed_force_returns_all(readings_dir: Path) -> None:
    pdf_path = readings_dir / "0.rag" / "paper1.pdf"

    with patch("agents.research.__main__.settings") as mock_settings:
        mock_settings.readings_dir = readings_dir

        key = _relative_key(pdf_path)
        manifest = {key: {"date": "2026-04-06", "hash": _pdf_hash(pdf_path), "note": "paper1"}}
        unprocessed = _find_unprocessed(manifest, force=True)

    assert len(unprocessed) == 3


def test_find_unprocessed_missing_dir(tmp_path: Path) -> None:
    with patch("agents.research.__main__.settings") as mock_settings:
        mock_settings.readings_dir = tmp_path / "nonexistent"
        unprocessed = _find_unprocessed({})

    assert unprocessed == []


# --- Relative key ---


def test_relative_key_within_readings(readings_dir: Path) -> None:
    pdf = readings_dir / "0.rag" / "paper1.pdf"
    with patch("agents.research.__main__.settings") as mock_settings:
        mock_settings.readings_dir = readings_dir
        key = _relative_key(pdf)
    assert key == "0.rag/paper1.pdf"


def test_relative_key_outside_readings(tmp_path: Path) -> None:
    pdf = tmp_path / "random" / "file.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"data")
    with patch("agents.research.__main__.settings") as mock_settings:
        mock_settings.readings_dir = tmp_path / "different_dir"
        key = _relative_key(pdf)
    assert key == str(pdf.resolve())
