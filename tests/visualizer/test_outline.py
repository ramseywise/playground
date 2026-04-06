from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agents.visualizer.outline import generate_outline
from agents.visualizer.models import DeckIntake, DeckOutline


def _fake_response(text: str) -> MagicMock:
    mock = MagicMock()
    mock.content = [MagicMock(text=text)]
    return mock


def test_generate_outline_parses_json() -> None:
    """Claude returns JSON -> DeckOutline model."""
    fake_json = json.dumps({
        "title": "Test Deck",
        "slides": [
            {
                "number": 1,
                "title": "Intro",
                "type": "narrative",
                "talking_points": ["point one", "point two"],
                "speaker_note": "Set the stage",
            }
        ],
    })
    with patch("agents.visualizer.outline.create_client") as mock_create:
        mock_create.return_value.messages.create.return_value = _fake_response(fake_json)
        intake = DeckIntake(goal="test", audience="eng", tone="casual")
        outline = generate_outline(intake, "claude-sonnet-4-6")

    assert outline.title == "Test Deck"
    assert len(outline.slides) == 1
    assert outline.slides[0].title == "Intro"
    assert outline.slides[0].type == "narrative"


def test_generate_outline_strips_fences() -> None:
    """JSON wrapped in ```json fences is still parsed correctly."""
    raw = json.dumps({
        "title": "Fenced Deck",
        "slides": [
            {
                "number": 1,
                "title": "Slide 1",
                "type": "concept",
                "talking_points": ["a"],
                "speaker_note": "note",
            }
        ],
    })
    fenced = f"```json\n{raw}\n```"
    with patch("agents.visualizer.outline.create_client") as mock_create:
        mock_create.return_value.messages.create.return_value = _fake_response(fenced)
        intake = DeckIntake(goal="test", audience="eng", tone="formal")
        outline = generate_outline(intake, "claude-sonnet-4-6")

    assert outline.title == "Fenced Deck"


def test_deck_outline_model_roundtrip() -> None:
    """DeckOutline serializes and deserializes correctly."""
    data = {
        "title": "Roundtrip",
        "slides": [
            {
                "number": 1,
                "title": "S1",
                "type": "architecture",
                "talking_points": ["a", "b"],
                "speaker_note": "note",
            },
            {
                "number": 2,
                "title": "S2",
                "type": "data",
                "talking_points": ["c"],
                "speaker_note": "note2",
            },
        ],
    }
    outline = DeckOutline(**data)
    assert len(outline.slides) == 2
    dumped = outline.model_dump()
    assert dumped["title"] == "Roundtrip"
    assert dumped["slides"][1]["type"] == "data"
