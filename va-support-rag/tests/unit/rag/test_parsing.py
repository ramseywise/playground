"""Unit tests for LLM JSON parsing utility."""

from __future__ import annotations

import pytest

from core.parsing import parse_json_safe


class TestParseJsonSafe:
    def test_plain_json_object(self) -> None:
        assert parse_json_safe('{"key": "value"}') == {"key": "value"}

    def test_plain_json_array(self) -> None:
        assert parse_json_safe("[1, 2, 3]") == [1, 2, 3]

    def test_json_in_triple_backtick_fence(self) -> None:
        text = '```\n{"a": 1}\n```'
        assert parse_json_safe(text) == {"a": 1}

    def test_json_in_named_fence(self) -> None:
        text = '```json\n{"x": true}\n```'
        assert parse_json_safe(text) == {"x": True}

    def test_leading_and_trailing_whitespace_stripped(self) -> None:
        assert parse_json_safe('   {"n": 42}   ') == {"n": 42}

    def test_invalid_json_returns_none(self) -> None:
        assert parse_json_safe("this is not json") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_json_safe("") is None

    def test_malformed_fence_no_closing_backticks(self) -> None:
        # Opens a fence but never closes — lines[1:] is used, still attempts parse
        text = '```json\n{"partial": true}'
        result = parse_json_safe(text)
        assert result == {"partial": True}

    def test_nested_json(self) -> None:
        text = '{"outer": {"inner": [1, 2]}}'
        assert parse_json_safe(text) == {"outer": {"inner": [1, 2]}}

    def test_json_number(self) -> None:
        assert parse_json_safe("3.14") == pytest.approx(3.14)

    def test_json_null(self) -> None:
        # null parses to Python None — can't distinguish from parse failure,
        # but the function still returns None which is the correct value
        assert parse_json_safe("null") is None

    def test_json_boolean_true(self) -> None:
        assert parse_json_safe("true") is True

    def test_json_boolean_false(self) -> None:
        assert parse_json_safe("false") is False
