from __future__ import annotations

import pytest

from core.client import strip_json_fences, create_client
from unittest.mock import patch


def test_strip_json_fences_plain() -> None:
    """Plain JSON passes through unchanged."""
    assert strip_json_fences('{"a": 1}') == '{"a": 1}'


def test_strip_json_fences_with_json_tag() -> None:
    """```json ... ``` fences are stripped."""
    assert strip_json_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_json_fences_without_json_tag() -> None:
    """Plain ``` ... ``` fences are stripped."""
    assert strip_json_fences('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_json_fences_with_whitespace() -> None:
    """Leading/trailing whitespace is handled."""
    assert strip_json_fences('  ```json\n{"a": 1}\n```  ') == '{"a": 1}'


def test_strip_json_fences_no_fences() -> None:
    """Text without fences is just stripped of whitespace."""
    assert strip_json_fences("  hello  ") == "hello"


def test_create_client_raises_without_key() -> None:
    """create_client raises RuntimeError when API key is empty."""
    with patch("core.client.settings") as mock_settings:
        mock_settings.anthropic_api_key = ""
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            create_client()
