"""Token counting (tiktoken) — mocked for non-empty text to avoid network fetch in CI."""

from __future__ import annotations

import pytest

import rag.tokenization as tok
from rag.tokenization import count_tokens


def test_count_tokens_empty() -> None:
    assert count_tokens("") == 0


def test_count_tokens_short(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid downloading ``cl100k_base`` from Azure during unit tests."""
    fake = type(
        "Enc",
        (),
        {"encode": lambda self, t: [0] * max(1, len(t) // 2)},
    )()

    monkeypatch.setattr(tok, "_encoding", None, raising=False)
    monkeypatch.setattr(tok, "_encoding_cl100k", lambda: fake)

    assert count_tokens("abcd") == 2
