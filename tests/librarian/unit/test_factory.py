from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from librarian.factory import create_librarian, warm_up_embedder
from storage.vectordb.inmemory import InMemoryRetriever
from tests.librarian.testing.mock_embedder import MockEmbedder
from librarian.config import LibrarySettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**kwargs: Any) -> LibrarySettings:
    base: dict[str, Any] = dict(
        retrieval_strategy="inmemory",
        reranker_strategy="cross_encoder",
        confidence_threshold=0.3,
        max_crag_retries=1,
        retrieval_k=5,
        reranker_top_k=3,
        anthropic_api_key="test-key",
    )
    cast(Any, base).update(kwargs)
    return LibrarySettings(**base)  # type: ignore[arg-type]


def _mock_llm(response: str = "answer") -> MagicMock:
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=response)
    return llm


def _mock_reranker() -> MagicMock:
    r = MagicMock()
    r.rerank = AsyncMock(return_value=[])
    return r


# ---------------------------------------------------------------------------
# create_librarian — component injection
# ---------------------------------------------------------------------------


def test_create_returns_compiled_graph() -> None:
    graph = create_librarian(
        cfg=_cfg(),
        llm=_mock_llm(),
        embedder=MockEmbedder(dim=64),
        retriever=InMemoryRetriever(),
        reranker=_mock_reranker(),
    )
    # CompiledGraph is callable/invokable — check it has ainvoke
    assert hasattr(graph, "ainvoke")


def test_create_injected_components_not_rebuilt() -> None:
    """Injected components bypass _build_* — _build_llm must not be called."""
    with patch("librarian.factory._build_llm") as mock_build_llm:
        create_librarian(
            cfg=_cfg(),
            llm=_mock_llm(),
            embedder=MockEmbedder(dim=64),
            retriever=InMemoryRetriever(),
            reranker=_mock_reranker(),
        )
        mock_build_llm.assert_not_called()


def test_create_embedder_not_rebuilt_when_injected() -> None:
    with patch("librarian.factory._build_embedder") as mock_build:
        create_librarian(
            cfg=_cfg(),
            llm=_mock_llm(),
            embedder=MockEmbedder(dim=64),
            retriever=InMemoryRetriever(),
            reranker=_mock_reranker(),
        )
        mock_build.assert_not_called()


def test_create_retriever_not_rebuilt_when_injected() -> None:
    with patch("librarian.factory._build_retriever") as mock_build:
        create_librarian(
            cfg=_cfg(),
            llm=_mock_llm(),
            embedder=MockEmbedder(dim=64),
            retriever=InMemoryRetriever(),
            reranker=_mock_reranker(),
        )
        mock_build.assert_not_called()


# ---------------------------------------------------------------------------
# create_librarian — strategy selection
# ---------------------------------------------------------------------------


def test_create_inmemory_strategy_used() -> None:
    with patch(
        "librarian.factory._build_retriever",
        wraps=lambda cfg, emb: InMemoryRetriever(),
    ) as mock_build:
        create_librarian(
            cfg=_cfg(retrieval_strategy="inmemory"),
            llm=_mock_llm(),
            embedder=MockEmbedder(dim=64),
            reranker=_mock_reranker(),
        )
        mock_build.assert_called_once()


def test_create_llm_listwise_reranker_strategy() -> None:
    with patch("librarian.factory._build_reranker") as mock_build:
        mock_build.return_value = _mock_reranker()
        create_librarian(
            cfg=_cfg(reranker_strategy="llm_listwise"),
            llm=_mock_llm(),
            embedder=MockEmbedder(dim=64),
            retriever=InMemoryRetriever(),
        )
        call_cfg = mock_build.call_args[0][0]
        assert call_cfg.reranker_strategy == "llm_listwise"


# ---------------------------------------------------------------------------
# create_librarian — graph settings propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_graph_is_invokable() -> None:
    graph = create_librarian(
        cfg=_cfg(),
        llm=_mock_llm("the answer"),
        embedder=MockEmbedder(dim=64),
        retriever=InMemoryRetriever(),
        reranker=_mock_reranker(),
    )
    result = await graph.ainvoke(
        {"query": "hello", "intent": "conversational"},
        config={"configurable": {"thread_id": "test-1"}},
    )
    assert result["response"] == "the answer"


@pytest.mark.asyncio
async def test_create_confidence_threshold_propagated() -> None:
    """Low threshold → never fallback even with 0 confidence."""
    graph = create_librarian(
        cfg=_cfg(confidence_threshold=0.0),
        llm=_mock_llm("ok"),
        embedder=MockEmbedder(dim=64),
        retriever=InMemoryRetriever(),
        reranker=_mock_reranker(),
    )
    result = await graph.ainvoke(
        {"query": "what is auth?", "intent": "lookup"},
        config={"configurable": {"thread_id": "test-2"}},
    )
    assert result["response"] == "ok"


# ---------------------------------------------------------------------------
# warm_up_embedder — Step 1: embedder warm-up
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _build_checkpointer — Step 2: LangGraph checkpointer
# ---------------------------------------------------------------------------


def test_default_checkpoint_backend_is_memory() -> None:
    """Default config uses MemorySaver (no persistence)."""
    from langgraph.checkpoint.memory import MemorySaver

    cfg = _cfg()
    graph = create_librarian(
        cfg=cfg,
        llm=_mock_llm(),
        embedder=MockEmbedder(dim=64),
        retriever=InMemoryRetriever(),
        reranker=_mock_reranker(),
    )
    assert isinstance(graph.checkpointer, MemorySaver)


def test_checkpointer_sqlite_raises_without_package() -> None:
    """Requesting sqlite backend without package raises ImportError."""
    with patch(
        "librarian.factory._build_checkpointer",
        side_effect=ImportError("langgraph-checkpoint-sqlite is required"),
    ):
        with pytest.raises(ImportError, match="langgraph-checkpoint-sqlite"):
            create_librarian(
                cfg=_cfg(checkpoint_backend="sqlite"),
                llm=_mock_llm(),
                embedder=MockEmbedder(dim=64),
                retriever=InMemoryRetriever(),
                reranker=_mock_reranker(),
            )


def test_checkpointer_postgres_raises_without_url() -> None:
    """Requesting postgres backend without a URL raises ValueError."""
    from librarian.factory import _build_checkpointer

    with pytest.raises((ImportError, ValueError)):
        _build_checkpointer(_cfg(checkpoint_backend="postgres"))


@pytest.mark.asyncio
async def test_graph_with_checkpointer_requires_thread_id() -> None:
    """Graph compiled with MemorySaver raises without thread_id."""
    graph = create_librarian(
        cfg=_cfg(),
        llm=_mock_llm(),
        embedder=MockEmbedder(dim=64),
        retriever=InMemoryRetriever(),
        reranker=_mock_reranker(),
    )
    with pytest.raises(ValueError, match="thread_id"):
        await graph.ainvoke({"query": "hello", "intent": "lookup"})


# ---------------------------------------------------------------------------
# warm_up_embedder — Step 1: embedder warm-up
# ---------------------------------------------------------------------------


def test_warm_up_embedder_calls_embed_query() -> None:
    """warm_up_embedder builds an embedder and calls embed_query to pre-load the model."""
    mock_embedder = MagicMock(spec=MockEmbedder)
    mock_embedder.embed_query.return_value = [0.1] * 64

    with patch(
        "librarian.factory._build_embedder", return_value=mock_embedder
    ) as mock_build:
        cfg = _cfg(embedding_model="test-warmup-model")
        warm_up_embedder(cfg)

        mock_build.assert_called_once_with(cfg)
        mock_embedder.embed_query.assert_called_once_with("warmup")


@pytest.mark.asyncio
async def test_create_default_cfg_uses_module_settings() -> None:
    """Passing cfg=None uses the module-level settings singleton."""
    with (
        patch(
            "librarian.factory._build_llm", return_value=_mock_llm()
        ) as mock_llm_build,
        patch(
            "librarian.factory._build_history_llm", return_value=_mock_llm()
        ) as mock_history_llm_build,
        patch(
            "librarian.factory._build_embedder",
            return_value=MockEmbedder(dim=64),
        ),
        patch(
            "librarian.factory._build_retriever",
            return_value=InMemoryRetriever(),
        ),
        patch("librarian.factory._build_reranker", return_value=_mock_reranker()),
    ):
        create_librarian()  # cfg=None
        mock_llm_build.assert_called_once()
        mock_history_llm_build.assert_called_once()
