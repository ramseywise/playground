from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.research.extractor import extract_pages, extract_toc, get_page_count

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


# --- Unit tests (mocked subprocess) ---


def test_get_page_count_parses_pdfinfo_output() -> None:
    mock_result = MagicMock()
    mock_result.stdout = "Title: Test\nPages: 42\nFile size: 1234 bytes\n"
    with patch("agents.research.extractor.subprocess.run", return_value=mock_result):
        assert get_page_count(Path("fake.pdf")) == 42


def test_get_page_count_raises_if_no_pages_line() -> None:
    mock_result = MagicMock()
    mock_result.stdout = "Title: Test\nAuthor: Someone\n"
    with patch("agents.research.extractor.subprocess.run", return_value=mock_result):
        with pytest.raises(ValueError, match="Could not parse page count"):
            get_page_count(Path("fake.pdf"))


def test_extract_pages_returns_stdout() -> None:
    mock_result = MagicMock()
    mock_result.stdout = "This is the extracted text."
    with patch("agents.research.extractor.subprocess.run", return_value=mock_result):
        text = extract_pages(Path("fake.pdf"), 1, 5)
    assert text == "This is the extracted text."


def test_extract_toc_calls_extract_pages_with_first_10() -> None:
    with patch("agents.research.extractor.extract_pages", return_value="toc text") as mock_ep:
        result = extract_toc(Path("fake.pdf"))
    mock_ep.assert_called_once_with(Path("fake.pdf"), 1, 10)
    assert result == "toc text"


def test_extract_toc_custom_pages() -> None:
    with patch("agents.research.extractor.extract_pages", return_value="toc") as mock_ep:
        extract_toc(Path("fake.pdf"), toc_pages=5)
    mock_ep.assert_called_once_with(Path("fake.pdf"), 1, 5)


# --- Integration tests (real binary) ---


@pytest.mark.integration
def test_get_page_count_real_pdf() -> None:
    count = get_page_count(FIXTURE_PDF)
    assert count == 12  # 1301.3781v3.pdf has 12 pages (pdfinfo confirmed)


@pytest.mark.integration
def test_extract_pages_real_pdf_returns_text() -> None:
    text = extract_pages(FIXTURE_PDF, 1, 2)
    assert len(text) > 100
    assert isinstance(text, str)


@pytest.mark.integration
def test_extract_toc_real_pdf_returns_text() -> None:
    text = extract_toc(FIXTURE_PDF)
    assert len(text) > 50
