"""Unit tests for retrieval scoring primitives."""

from __future__ import annotations


import pytest

from rag.retrieval.scoring import cosine_similarity, term_overlap


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_one(self) -> None:
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_a_returns_zero(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_zero_vector_b_returns_zero(self) -> None:
        assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_both_zero_vectors_return_zero(self) -> None:
        assert cosine_similarity([0.0], [0.0]) == 0.0

    def test_known_similarity(self) -> None:
        # [3, 4] vs [4, 3]: dot=24, |a|=5, |b|=5 → 24/25 = 0.96
        result = cosine_similarity([3.0, 4.0], [4.0, 3.0])
        assert result == pytest.approx(0.96)

    def test_single_element_vectors(self) -> None:
        assert cosine_similarity([2.0], [5.0]) == pytest.approx(1.0)

    def test_result_bounded_between_neg_one_and_one(self) -> None:
        import random

        rng = random.Random(42)
        for _ in range(20):
            a = [rng.uniform(-1, 1) for _ in range(8)]
            b = [rng.uniform(-1, 1) for _ in range(8)]
            result = cosine_similarity(a, b)
            assert -1.0 - 1e-9 <= result <= 1.0 + 1e-9


class TestTermOverlap:
    def test_full_overlap(self) -> None:
        assert term_overlap("hello world", "hello world foo") == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert term_overlap("cat dog", "fish bird") == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        # 1 of 2 query terms present → 0.5
        assert term_overlap("cat dog", "cat fish") == pytest.approx(0.5)

    def test_empty_query_returns_zero(self) -> None:
        assert term_overlap("", "some text here") == 0.0

    def test_empty_text_no_overlap(self) -> None:
        assert term_overlap("hello", "") == pytest.approx(0.0)

    def test_case_insensitive(self) -> None:
        assert term_overlap("Hello WORLD", "hello world") == pytest.approx(1.0)

    def test_duplicate_query_terms_treated_as_set(self) -> None:
        # "cat cat" → {"cat"}, "cat dog" → {"cat", "dog"}: overlap={"cat"}/{"cat"}=1.0
        assert term_overlap("cat cat", "cat dog") == pytest.approx(1.0)
