"""Tests for viz classifier — two-pass prompting and model construction."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agents.visualizer.models import ImageConcept, SlideContent, VizPrompt
from agents.visualizer.providers import PollinationsProvider
from agents.visualizer.viz_classifier import (
    _generate_scene_description,
    _translate_to_image_prompt,
    classify_slides,
)


def _fake_response(text: str) -> MagicMock:
    mock = MagicMock()
    mock.content = [MagicMock(text=text)]
    return mock


# --- Model construction tests ---


def test_viz_prompt_skip_image() -> None:
    """VizPrompt with skip_image=True has no URL."""
    vp = VizPrompt(slide_number=1, viz_type="data", skip_image=True)
    assert vp.skip_image is True
    assert vp.pollinations_url is None


def test_viz_prompt_with_url() -> None:
    """VizPrompt with an image has a pollinations URL."""
    vp = VizPrompt(
        slide_number=2,
        viz_type="concept",
        skip_image=False,
        pollinations_url="https://image.pollinations.ai/prompt/test",
        filled_prompt="test prompt",
    )
    assert vp.skip_image is False
    assert "pollinations" in vp.pollinations_url


def test_pollinations_build_url_encodes_prompt() -> None:
    """PollinationsProvider._build_url properly URL-encodes the prompt text."""
    provider = PollinationsProvider()
    url = provider._build_url("hello world", width=800, height=600)
    assert "hello%20world" in url
    assert "width=800" in url
    assert "height=600" in url
    assert "nologo=true" in url
    assert "model=flux" in url


def test_image_concept_model() -> None:
    """ImageConcept model construction."""
    concept = ImageConcept(
        label="Neural Network Flow",
        viz_type="architecture",
        description="Diagram of data flowing through layers",
        rationale="Matches the technical audience",
    )
    assert concept.label == "Neural Network Flow"
    assert concept.pollinations_url is None


def test_slide_content_no_image_types() -> None:
    """SlideContent for data/code_demo/team types should have no image_brief."""
    for slide_type in ("data", "code_demo", "team"):
        sc = SlideContent(
            slide_number=1,
            slide_type=slide_type,
            headline="Test",
            body=["bullet"],
            speaker_note="note",
            image_brief=None,
        )
        assert sc.image_brief is None


# --- Two-pass prompting tests ---


def test_generate_scene_description() -> None:
    """Pass 1: scene description returns structured JSON."""
    scene_json = json.dumps({
        "scene": "A glowing neural network with data flowing through layers",
        "key_elements": ["neural network", "data flow", "glowing nodes"],
        "mood": "futuristic",
        "color_palette": ["cyan", "dark navy", "white"],
    })
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_response(scene_json)

    result = _generate_scene_description(
        mock_client, "claude-sonnet-4-6", "Slide context here", "Deck style here"
    )

    assert result["scene"] == "A glowing neural network with data flowing through layers"
    assert len(result["key_elements"]) == 3
    assert result["mood"] == "futuristic"
    # Verify deck style context was passed to the prompt
    call_args = mock_client.messages.create.call_args
    user_msg = call_args[1]["messages"][0]["content"]
    assert "Deck style here" in user_msg


def test_translate_to_image_prompt() -> None:
    """Pass 2: scene translates to filled template prompt."""
    filled_json = json.dumps({
        "concept_name": "machine learning",
        "metaphor_description": "glowing neural pathways",
        "emotional_tone": "innovative and powerful",
        "color_mood": "cyan and navy",
        "composition_style": "centered radial",
    })
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_response(filled_json)

    scene = {
        "scene": "A glowing neural network",
        "key_elements": ["nodes", "connections"],
        "mood": "futuristic",
        "color_palette": ["cyan", "navy"],
    }
    template = "abstract visual metaphor for {concept_name}: {metaphor_description}, conveying {emotional_tone}, {color_mood} color palette, {composition_style} composition"
    style = "abstract geometric illustration"

    result = _translate_to_image_prompt(
        mock_client, "claude-sonnet-4-6", scene, template, style, "Use symbolism."
    )

    assert "machine learning" in result
    assert "glowing neural pathways" in result
    assert "abstract geometric illustration" in result


def test_classify_slides_two_pass(tmp_path: str) -> None:
    """classify_slides uses two LLM calls per image slide (scene + prompt)."""
    scene_json = json.dumps({
        "scene": "test scene",
        "key_elements": ["element"],
        "mood": "calm",
        "color_palette": ["blue"],
    })
    filled_json = json.dumps({
        "concept_name": "test",
        "metaphor_description": "flowing water",
        "emotional_tone": "serene",
        "color_mood": "ocean blue",
        "composition_style": "horizontal flow",
    })

    mock_client = MagicMock()
    # First call = scene description, second call = prompt translation
    mock_client.messages.create.side_effect = [
        _fake_response(scene_json),
        _fake_response(filled_json),
    ]

    slides = [
        SlideContent(
            slide_number=1,
            slide_type="concept",
            headline="Test Slide",
            body=["point one"],
            speaker_note="note",
            image_brief="A flowing visual metaphor for data processing",
        ),
    ]

    with patch("agents.visualizer.viz_classifier.create_client", return_value=mock_client):
        result = classify_slides(slides, "claude-sonnet-4-6", "deck style context")

    assert len(result) == 1
    assert result[0].skip_image is False
    assert result[0].filled_prompt is not None
    # Two LLM calls: scene + prompt
    assert mock_client.messages.create.call_count == 2


def test_classify_slides_skips_no_image_types() -> None:
    """Slides with skip_image types are not processed."""
    slides = [
        SlideContent(
            slide_number=1,
            slide_type="data",
            headline="Data Slide",
            body=["numbers"],
            speaker_note="note",
            image_brief=None,
        ),
    ]

    mock_client = MagicMock()
    with patch("agents.visualizer.viz_classifier.create_client", return_value=mock_client):
        result = classify_slides(slides, "claude-sonnet-4-6")

    assert len(result) == 1
    assert result[0].skip_image is True
    mock_client.messages.create.assert_not_called()


def test_classify_slides_deck_style_propagates() -> None:
    """Deck style context appears in the scene description prompt."""
    scene_json = json.dumps({
        "scene": "test", "key_elements": [], "mood": "calm", "color_palette": [],
    })
    filled_json = json.dumps({"components": "a", "relationships": "b",
                               "infrastructure_type": "c", "scale": "d", "detail_focus": "e"})

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _fake_response(scene_json),
        _fake_response(filled_json),
    ]

    slides = [
        SlideContent(
            slide_number=1, slide_type="architecture", headline="Arch",
            body=["p"], speaker_note="n",
            image_brief="System diagram showing microservices",
        ),
    ]

    with patch("agents.visualizer.viz_classifier.create_client", return_value=mock_client):
        classify_slides(slides, "claude-sonnet-4-6", "Use dark theme with cyan accents")

    # First call is scene description — check deck style was passed
    scene_call = mock_client.messages.create.call_args_list[0]
    user_msg = scene_call[1]["messages"][0]["content"]
    assert "Use dark theme with cyan accents" in user_msg
