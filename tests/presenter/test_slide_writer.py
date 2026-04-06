"""Tests for slide content generation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agents.presenter.models import DeckIntake, DeckOutline, SlideContent
from agents.presenter.slide_writer import SLIDE_SYSTEM, generate_slide_content


def _fake_response(text: str) -> MagicMock:
    mock = MagicMock()
    mock.content = [MagicMock(text=text)]
    return mock


def test_generate_slide_content_parses_json() -> None:
    """SlideContent is correctly parsed from Claude JSON response."""
    fake_json = json.dumps({
        "headline": "Test Headline",
        "body": ["bullet one", "bullet two"],
        "speaker_note": "Speaker note here.",
        "image_brief": "A dramatic scene showing data flowing through a pipeline.",
    })

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_response(fake_json)

    outline = DeckOutline(
        title="Test Deck",
        slides=[{
            "number": 1,
            "title": "Intro",
            "type": "narrative",
            "talking_points": ["point one"],
            "speaker_note": "Set the stage",
        }],
    )
    intake = DeckIntake(goal="test goal", audience="engineers", tone="professional")

    with patch("agents.presenter.slide_writer.create_client", return_value=mock_client):
        result = generate_slide_content(outline, intake, "claude-sonnet-4-6")

    assert len(result) == 1
    assert result[0].headline == "Test Headline"
    assert len(result[0].body) == 2
    assert result[0].image_brief is not None
    assert result[0].slide_number == 1
    assert result[0].slide_type == "narrative"


def test_slide_system_prompt_requests_rich_image_brief() -> None:
    """SLIDE_SYSTEM instructs Claude to produce detailed image briefs."""
    assert "2-3 sentences" in SLIDE_SYSTEM
    assert "cinematic" in SLIDE_SYSTEM
    assert "concrete visual details" in SLIDE_SYSTEM


def test_no_image_types_get_null_brief() -> None:
    """Data/code_demo/team slides request null image_brief."""
    fake_json = json.dumps({
        "headline": "Data Slide",
        "body": ["stat one"],
        "speaker_note": "Show the numbers.",
        "image_brief": None,
    })

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_response(fake_json)

    outline = DeckOutline(
        title="Test Deck",
        slides=[{
            "number": 1,
            "title": "Metrics",
            "type": "data",
            "talking_points": ["growth"],
            "speaker_note": "Show metrics",
        }],
    )
    intake = DeckIntake(goal="test", audience="eng", tone="formal")

    with patch("agents.presenter.slide_writer.create_client", return_value=mock_client):
        result = generate_slide_content(outline, intake, "claude-sonnet-4-6")

    assert result[0].image_brief is None
