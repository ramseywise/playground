"""Unit tests for contracts — locale utility."""

from __future__ import annotations

from orchestrator.langgraph.schemas import SUPPORTED_LOCALES, locale_to_language


class TestLocaleToLanguage:
    def test_danish(self) -> None:
        assert locale_to_language("da") == "Danish"

    def test_german(self) -> None:
        assert locale_to_language("de") == "German"

    def test_french(self) -> None:
        assert locale_to_language("fr") == "French"

    def test_english(self) -> None:
        assert locale_to_language("en") == "English"

    def test_none_falls_back_to_english(self) -> None:
        assert locale_to_language(None) == "English"

    def test_unknown_locale_falls_back_to_english(self) -> None:
        assert locale_to_language("xx") == "English"

    def test_uppercase_normalized(self) -> None:
        assert locale_to_language("DA") == "Danish"
        assert locale_to_language("FR") == "French"

    def test_region_subtag_stripped(self) -> None:
        # "da-DK" → "da" → "Danish"
        assert locale_to_language("da-DK") == "Danish"
        assert locale_to_language("de-AT") == "German"
        assert locale_to_language("fr-BE") == "French"

    def test_all_supported_locales_resolve(self) -> None:
        for code, expected_language in SUPPORTED_LOCALES.items():
            assert locale_to_language(code) == expected_language
