"""Unit tests for chunking utility functions."""

from __future__ import annotations


from rag.preprocessing.base import ChunkerConfig
from rag.preprocessing.chunking.utils import (
    WORDS_PER_TOKEN,
    approx_tokens,
    hard_split_text,
    make_chunk,
    make_doc_id,
    recursive_split_by_separators,
    word_count,
)


class TestApproxTokens:
    def test_single_word(self) -> None:
        # 1 word / 0.75 = 1.33 → int → 1, max(1, 1) = 1
        assert approx_tokens("hello") == 1

    def test_empty_string(self) -> None:
        # 0 words → max(1, 0) = 1
        assert approx_tokens("") == 1

    def test_known_count(self) -> None:
        # 3 words / 0.75 = 4
        assert approx_tokens("one two three") == 4

    def test_scales_with_length(self) -> None:
        short = approx_tokens("a b c")
        long = approx_tokens("a b c d e f g h i j")
        assert long > short

    def test_uses_words_per_token_constant(self) -> None:
        words = "word " * 75
        expected = max(1, int(75 / WORDS_PER_TOKEN))
        assert approx_tokens(words.strip()) == expected


class TestWordCount:
    def test_empty_string(self) -> None:
        assert word_count("") == 0

    def test_single_word(self) -> None:
        assert word_count("hello") == 1

    def test_multiple_words(self) -> None:
        assert word_count("one two three four") == 4

    def test_extra_spaces_ignored(self) -> None:
        assert word_count("  a   b   c  ") == 3


class TestMakeDocId:
    def test_deterministic(self) -> None:
        id1 = make_doc_id("https://example.com", "intro")
        id2 = make_doc_id("https://example.com", "intro")
        assert id1 == id2

    def test_different_url_different_id(self) -> None:
        assert make_doc_id("https://a.com", None) != make_doc_id("https://b.com", None)

    def test_different_section_different_id(self) -> None:
        assert make_doc_id("https://x.com", "section-a") != make_doc_id(
            "https://x.com", "section-b"
        )

    def test_none_section_same_as_empty_string(self) -> None:
        assert make_doc_id("https://x.com", None) == make_doc_id("https://x.com", "")

    def test_output_is_16_hex_chars(self) -> None:
        doc_id = make_doc_id("https://example.com", "main")
        assert len(doc_id) == 16
        assert all(c in "0123456789abcdef" for c in doc_id)


class TestMakeChunk:
    def test_chunk_has_correct_text(self) -> None:
        chunk = make_chunk("hello world", "https://x.com", "Title", "Sec", "docid")
        assert chunk.text == "hello world"

    def test_chunk_metadata_populated(self) -> None:
        chunk = make_chunk("text", "https://x.com", "My Title", "Section 1", "docid")
        assert chunk.metadata.url == "https://x.com"
        assert chunk.metadata.title == "My Title"
        assert chunk.metadata.section == "Section 1"
        assert chunk.metadata.doc_id == "docid"

    def test_chunk_id_is_deterministic(self) -> None:
        c1 = make_chunk("text", "https://x.com", "t", "s", "d")
        c2 = make_chunk("text", "https://x.com", "t", "s", "d")
        assert c1.id == c2.id

    def test_different_text_different_id(self) -> None:
        c1 = make_chunk("foo", "https://x.com", "t", "s", "d")
        c2 = make_chunk("bar", "https://x.com", "t", "s", "d")
        assert c1.id != c2.id

    def test_chunk_id_is_20_hex_chars(self) -> None:
        chunk = make_chunk("some text", "https://x.com", "t", "s", "d")
        assert len(chunk.id) == 20
        assert all(c in "0123456789abcdef" for c in chunk.id)


class TestHardSplitText:
    def test_short_text_returns_single_chunk(self) -> None:
        config = ChunkerConfig(max_tokens=512, overlap_tokens=0, min_tokens=1)
        result = hard_split_text("short text", config)
        assert result == ["short text"]

    def test_empty_text_returns_empty_list(self) -> None:
        config = ChunkerConfig(max_tokens=512, overlap_tokens=0, min_tokens=1)
        result = hard_split_text("", config)
        assert result == []

    def test_long_text_splits_into_multiple_chunks(self) -> None:
        config = ChunkerConfig(max_tokens=10, overlap_tokens=0, min_tokens=1)
        text = " ".join([f"word{i}" for i in range(50)])
        result = hard_split_text(text, config)
        assert len(result) > 1

    def test_chunks_cover_all_words(self) -> None:
        config = ChunkerConfig(max_tokens=10, overlap_tokens=0, min_tokens=1)
        words = [f"w{i}" for i in range(30)]
        text = " ".join(words)
        result = hard_split_text(text, config)
        # All words should appear somewhere in the chunks
        all_words_in_chunks = set(" ".join(result).split())
        assert set(words).issubset(all_words_in_chunks)

    def test_min_tokens_filters_all_chunks_returns_empty(self) -> None:
        # When every chunk is below min_tokens the result is an empty list —
        # callers should not embed empty or sub-threshold text.
        config = ChunkerConfig(max_tokens=5, overlap_tokens=0, min_tokens=100)
        text = " ".join([f"w{i}" for i in range(4)])  # only 4 words → below min
        result = hard_split_text(text, config)
        assert result == []


class TestRecursiveSplitBySeparators:
    def test_short_text_not_split(self) -> None:
        result = recursive_split_by_separators(
            "hello world", max_tokens=100, overlap_tokens=0
        )
        assert result == ["hello world"]

    def test_paragraph_split(self) -> None:
        text = "paragraph one.\n\nparagraph two.\n\nparagraph three."
        result = recursive_split_by_separators(text, max_tokens=5, overlap_tokens=0)
        assert len(result) >= 2

    def test_overlap_carries_context(self) -> None:
        # Build text that forces a split and check overlap is present
        para_a = " ".join([f"word{i}" for i in range(10)])
        para_b = " ".join([f"other{i}" for i in range(10)])
        text = para_a + "\n\n" + para_b
        result = recursive_split_by_separators(text, max_tokens=12, overlap_tokens=3)
        if len(result) > 1:
            # The second chunk should carry some words from the tail of the first
            last_words_of_first = para_a.split()[-3:]
            second_chunk_words = result[1].split()
            overlap_found = any(w in second_chunk_words for w in last_words_of_first)
            assert overlap_found

    def test_word_level_fallback_for_no_separators(self) -> None:
        # Single long run of words with no paragraph/sentence separators
        text = " ".join([f"x{i}" for i in range(20)])
        result = recursive_split_by_separators(text, max_tokens=5, overlap_tokens=0)
        assert len(result) > 1
        # All words accounted for (with possible overlap)
        all_words = set(" ".join(result).split())
        assert set(text.split()).issubset(all_words)

    def test_single_word_not_infinite_loop(self) -> None:
        result = recursive_split_by_separators(
            "oneword", max_tokens=1, overlap_tokens=0
        )
        assert isinstance(result, list)
        assert len(result) >= 1
