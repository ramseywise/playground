from __future__ import annotations

from agents.presenter.viz_classifier import (
    VizPrompt,
    ImageConcept,
    _build_url,
)
from agents.presenter.slide_writer import SlideContent


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


def test_build_url_encodes_prompt() -> None:
    """_build_url properly URL-encodes the prompt text."""
    url = _build_url("hello world", width=800, height=600)
    assert "hello%20world" in url
    assert "width=800" in url
    assert "height=600" in url
    assert "nologo=true" in url


def test_image_concept_model() -> None:
    """ImageConcept model construction."""
    concept = ImageConcept(
        label="Neural Network Flow",
        viz_type="architecture",
        description="Diagram of data flowing through layers",
        rationale="Matches the technical audience",
    )
    assert concept.label == "Neural Network Flow"
    assert concept.pollinations_url is None  # not yet filled


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
