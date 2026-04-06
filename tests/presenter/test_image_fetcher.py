"""Tests for image fetcher — provider dispatch and parallel execution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from agents.presenter.image_fetcher import fetch_images_for_slides, fetch_single_image
from agents.presenter.models import VizPrompt


def _make_viz_prompt(
    slide_number: int,
    filled_prompt: str = "test prompt",
    skip: bool = False,
) -> VizPrompt:
    return VizPrompt(
        slide_number=slide_number,
        viz_type="concept",
        skip_image=skip,
        filled_prompt=None if skip else filled_prompt,
    )


def test_fetch_images_skips_skip_image(tmp_path: Path) -> None:
    """Slides with skip_image=True are not fetched."""
    prompts = [
        _make_viz_prompt(1, skip=True),
        _make_viz_prompt(2, skip=True),
    ]

    mock_provider = MagicMock()
    with patch("agents.presenter.image_fetcher.get_provider", return_value=mock_provider):
        result = fetch_images_for_slides(prompts, tmp_path, "test_deck")

    assert result == {}
    mock_provider.generate_image.assert_not_called()


def test_fetch_images_parallel(tmp_path: Path) -> None:
    """Multiple non-skipped slides are fetched (provider called for each)."""
    prompts = [
        _make_viz_prompt(1, "prompt one"),
        _make_viz_prompt(2, skip=True),
        _make_viz_prompt(3, "prompt three"),
        _make_viz_prompt(4, "prompt four"),
    ]

    mock_provider = MagicMock()
    mock_provider.generate_image.side_effect = lambda prompt, dest, w, h: dest

    with patch("agents.presenter.image_fetcher.get_provider", return_value=mock_provider):
        with patch("agents.presenter.image_fetcher.settings") as mock_settings:
            mock_settings.image_width = 1280
            mock_settings.image_height = 720
            result = fetch_images_for_slides(prompts, tmp_path, "test_deck")

    assert len(result) == 3
    assert 1 in result
    assert 3 in result
    assert 4 in result
    assert 2 not in result  # skipped
    assert mock_provider.generate_image.call_count == 3


def test_fetch_images_handles_failure(tmp_path: Path) -> None:
    """A failed fetch logs error but doesn't crash the batch."""
    prompts = [
        _make_viz_prompt(1, "prompt one"),
        _make_viz_prompt(2, "prompt two"),
    ]

    call_count = 0

    def side_effect(prompt: str, dest: Path, w: int, h: int) -> Path:
        nonlocal call_count
        call_count += 1
        if "one" in prompt:
            raise RuntimeError("network error")
        return dest

    mock_provider = MagicMock()
    mock_provider.generate_image.side_effect = side_effect

    with patch("agents.presenter.image_fetcher.get_provider", return_value=mock_provider):
        with patch("agents.presenter.image_fetcher.settings") as mock_settings:
            mock_settings.image_width = 1280
            mock_settings.image_height = 720
            result = fetch_images_for_slides(prompts, tmp_path, "test_deck")

    # Slide 1 failed, slide 2 succeeded
    assert 2 in result
    assert 1 not in result


def test_fetch_single_image(tmp_path: Path) -> None:
    """fetch_single_image dispatches to provider."""
    mock_provider = MagicMock()
    expected_dest = tmp_path / "output.png"
    mock_provider.generate_image.return_value = expected_dest

    with patch("agents.presenter.image_fetcher.get_provider", return_value=mock_provider):
        with patch("agents.presenter.image_fetcher.settings") as mock_settings:
            mock_settings.image_width = 1280
            mock_settings.image_height = 720
            result = fetch_single_image("a prompt", tmp_path, "output.png")

    assert result == expected_dest
    mock_provider.generate_image.assert_called_once_with(
        "a prompt", expected_dest, 1280, 720
    )
