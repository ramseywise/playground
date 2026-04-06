"""Verify visualizer-related Settings fields exist and have correct defaults."""

from __future__ import annotations

from pathlib import Path

from agents.shared.config import Settings


def test_visualizer_settings_defaults() -> None:
    """All visualizer config fields are present with expected defaults."""
    s = Settings(anthropic_api_key="test-key")
    assert s.image_provider == "pollinations"
    assert s.pollinations_model == "flux"
    assert s.pollinations_seed is None
    assert s.pollinations_enhance is False
    assert s.replicate_api_token == ""
    assert s.viz_output_dir == Path("output")
    assert s.image_width == 1280
    assert s.image_height == 720
    assert s.viz_audience == "mixed technical and product team"
    assert s.viz_model == "claude-sonnet-4-6"


def test_visualizer_settings_override() -> None:
    """Visualizer fields can be overridden via constructor (simulating env vars)."""
    s = Settings(
        anthropic_api_key="test-key",
        image_provider="replicate",
        pollinations_model="turbo",
        pollinations_seed=42,
        pollinations_enhance=True,
        image_width=1920,
        image_height=1080,
    )
    assert s.image_provider == "replicate"
    assert s.pollinations_model == "turbo"
    assert s.pollinations_seed == 42
    assert s.pollinations_enhance is True
    assert s.image_width == 1920
    assert s.image_height == 1080
