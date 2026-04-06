"""Tests for image generation providers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.presenter.providers import (
    PollinationsProvider,
    ReplicateProvider,
    get_provider,
)


# --- PollinationsProvider ---


def test_pollinations_build_url_default_params() -> None:
    """Default URL includes model=flux and nologo=true."""
    provider = PollinationsProvider()
    url = provider._build_url("hello world", 1280, 720)
    assert "hello%20world" in url
    assert "model=flux" in url
    assert "nologo=true" in url
    assert "width=1280" in url
    assert "height=720" in url
    assert "seed=" not in url
    assert "enhance=" not in url


def test_pollinations_build_url_with_seed() -> None:
    """URL includes seed when configured."""
    provider = PollinationsProvider(seed=42)
    url = provider._build_url("test", 800, 600)
    assert "seed=42" in url


def test_pollinations_build_url_with_enhance() -> None:
    """URL includes enhance=true when configured."""
    provider = PollinationsProvider(enhance=True)
    url = provider._build_url("test", 800, 600)
    assert "enhance=true" in url


def test_pollinations_build_url_all_params() -> None:
    """URL includes all params when fully configured."""
    provider = PollinationsProvider(model="turbo", seed=123, enhance=True)
    url = provider._build_url("a prompt", 1920, 1080)
    assert "model=turbo" in url
    assert "seed=123" in url
    assert "enhance=true" in url
    assert "width=1920" in url
    assert "height=1080" in url


def test_pollinations_generate_image(tmp_path: Path) -> None:
    """generate_image fetches URL and writes file."""
    provider = PollinationsProvider()
    dest = tmp_path / "test.png"
    fake_content = b"fake-png-data"

    mock_response = MagicMock()
    mock_response.content = fake_content
    mock_response.raise_for_status = MagicMock()

    with patch("agents.presenter.providers.httpx.Client") as mock_client_cls:
        ctx = MagicMock()
        ctx.get.return_value = mock_response
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=ctx)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = provider.generate_image("test prompt", dest, 1280, 720)

    assert result == dest
    assert dest.read_bytes() == fake_content


# --- ReplicateProvider ---


def test_replicate_provider_import_error() -> None:
    """ReplicateProvider raises RuntimeError when replicate is not installed."""
    with patch.dict("sys.modules", {"replicate": None}):
        with pytest.raises(RuntimeError, match="uv add replicate"):
            ReplicateProvider(api_token="test-token")


# --- get_provider factory ---


def test_get_provider_default_pollinations() -> None:
    """Default provider is PollinationsProvider."""
    with patch("agents.presenter.providers.settings") as mock_settings:
        mock_settings.image_provider = "pollinations"
        mock_settings.pollinations_model = "flux"
        mock_settings.pollinations_seed = None
        mock_settings.pollinations_enhance = False
        provider = get_provider()
    assert isinstance(provider, PollinationsProvider)
    assert provider.model == "flux"


def test_get_provider_pollinations_with_params() -> None:
    """PollinationsProvider picks up config params."""
    with patch("agents.presenter.providers.settings") as mock_settings:
        mock_settings.image_provider = "pollinations"
        mock_settings.pollinations_model = "turbo"
        mock_settings.pollinations_seed = 42
        mock_settings.pollinations_enhance = True
        provider = get_provider()
    assert isinstance(provider, PollinationsProvider)
    assert provider.model == "turbo"
    assert provider.seed == 42
    assert provider.enhance is True


def test_get_provider_replicate_no_token() -> None:
    """Replicate provider without token raises RuntimeError."""
    with patch("agents.presenter.providers.settings") as mock_settings:
        mock_settings.image_provider = "replicate"
        mock_settings.replicate_api_token = ""
        with pytest.raises(RuntimeError, match="REPLICATE_API_TOKEN"):
            get_provider()
