"""Tests for generation/context.py (source diversity + XML formatting)."""

from __future__ import annotations

from librarian.generation.context import (
    deduplicate_by_source,
    format_as_xml_context,
)
from librarian.schemas.chunks import Chunk, ChunkMetadata, RankedChunk


def _ranked(
    id_: str,
    url: str = "https://a.com",
    score: float = 0.8,
    title: str = "Title",
    text: str = "Some content",
) -> RankedChunk:
    return RankedChunk(
        chunk=Chunk(
            id=id_,
            text=text,
            metadata=ChunkMetadata(url=url, title=title, doc_id="d1"),
        ),
        relevance_score=score,
        rank=1,
    )


class TestDeduplicateBySource:
    def test_empty_input(self) -> None:
        assert deduplicate_by_source([]) == []

    def test_all_different_sources_kept(self) -> None:
        chunks = [
            _ranked("a", url="https://a.com"),
            _ranked("b", url="https://b.com"),
            _ranked("c", url="https://c.com"),
        ]
        result = deduplicate_by_source(chunks)
        assert len(result) == 3

    def test_limits_per_source(self) -> None:
        chunks = [
            _ranked("a1", url="https://a.com"),
            _ranked("a2", url="https://a.com"),
            _ranked("a3", url="https://a.com"),
        ]
        result = deduplicate_by_source(chunks, max_per_source=2)
        assert len(result) == 2
        assert result[0].chunk.id == "a1"
        assert result[1].chunk.id == "a2"

    def test_custom_max_per_source(self) -> None:
        chunks = [_ranked(f"a{i}", url="https://a.com") for i in range(5)]
        result = deduplicate_by_source(chunks, max_per_source=1)
        assert len(result) == 1

    def test_preserves_order(self) -> None:
        chunks = [
            _ranked("a1", url="https://a.com"),
            _ranked("b1", url="https://b.com"),
            _ranked("a2", url="https://a.com"),
            _ranked("a3", url="https://a.com"),
        ]
        result = deduplicate_by_source(chunks, max_per_source=2)
        assert [r.chunk.id for r in result] == ["a1", "b1", "a2"]

    def test_mixed_sources_diverse(self) -> None:
        chunks = [
            _ranked("a1", url="https://a.com"),
            _ranked("a2", url="https://a.com"),
            _ranked("a3", url="https://a.com"),
            _ranked("b1", url="https://b.com"),
        ]
        result = deduplicate_by_source(chunks, max_per_source=2)
        urls = [r.chunk.metadata.url for r in result]
        assert urls.count("https://a.com") == 2
        assert urls.count("https://b.com") == 1


class TestFormatAsXmlContext:
    def test_empty_input(self) -> None:
        assert format_as_xml_context([]) == ""

    def test_single_chunk(self) -> None:
        chunks = [_ranked("a", title="My Doc", text="Hello world", score=0.95)]
        result = format_as_xml_context(chunks)
        assert '<document index="1"' in result
        assert "relevance=" in result
        assert "<title>My Doc</title>" in result
        assert "<content>\nHello world\n</content>" in result

    def test_multiple_chunks_separated(self) -> None:
        chunks = [_ranked("a"), _ranked("b")]
        result = format_as_xml_context(chunks)
        assert '<document index="1"' in result
        assert '<document index="2"' in result
        assert "\n\n" in result

    def test_includes_url(self) -> None:
        chunks = [_ranked("a", url="https://example.com/page")]
        result = format_as_xml_context(chunks)
        assert "<url>https://example.com/page</url>" in result

    def test_includes_relevance_score(self) -> None:
        chunks = [_ranked("a", score=0.73)]
        result = format_as_xml_context(chunks)
        assert 'relevance="0.73"' in result

    def test_last_updated_included_when_present(self) -> None:
        rc = _ranked("a")
        rc.chunk.metadata.last_updated = "2025-01-15"
        result = format_as_xml_context([rc])
        assert "<updated>2025-01-15</updated>" in result

    def test_last_updated_omitted_when_none(self) -> None:
        chunks = [_ranked("a")]
        result = format_as_xml_context(chunks)
        assert "<updated>" not in result
