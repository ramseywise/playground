"""Confidence routing — pure logic, no LLM."""

from orchestrator.langgraph.policies.confidence_routing import (
    decide_after_retrieval,
    decide_after_rerank,
    decide_qa_branch,
    retrieval_signal,
)
from orchestrator.langgraph.routing import (
    parse_retrieval_gate_decision,
    parse_rerank_gate_decision,
    route_after_qa_policy_retrieval,
    route_after_qa_policy_rerank,
    route_after_qa_retrieval_gate,
    route_after_qa_rerank_gate,
)
from orchestrator.langgraph.schemas.state import GraphState
from rag.schemas.chunks import Chunk, ChunkMetadata, GradedChunk, RankedChunk


def _ranked(score: float) -> RankedChunk:
    return RankedChunk(
        chunk=Chunk(id="1", text="x", metadata=ChunkMetadata()),
        relevance_score=score,
        rank=1,
    )


def _graded(score: float) -> GradedChunk:
    return GradedChunk(
        chunk=Chunk(id="1", text="x", metadata=ChunkMetadata()),
        score=score,
        relevant=True,
    )


def test_decide_answer_when_confident() -> None:
    branch, reason = decide_qa_branch(
        error=None,
        reranked_chunks=[_ranked(0.9)],
        confidence_score=0.9,
        threshold=0.25,
    )
    assert branch == "answer"
    assert reason is None


def test_decide_escalate_on_error() -> None:
    branch, reason = decide_qa_branch(
        error="boom",
        reranked_chunks=[],
        confidence_score=0.0,
        threshold=0.25,
    )
    assert branch == "escalate"
    assert reason == "boom"


def test_decide_escalate_empty_chunks() -> None:
    branch, reason = decide_qa_branch(
        error=None,
        reranked_chunks=[],
        confidence_score=0.0,
        threshold=0.25,
    )
    assert branch == "escalate"
    assert reason == "no_retrieval_results"


def test_decide_escalate_low_confidence() -> None:
    branch, reason = decide_qa_branch(
        error=None,
        reranked_chunks=[_ranked(0.1)],
        confidence_score=0.1,
        threshold=0.25,
    )
    assert branch == "escalate"
    assert reason == "low_confidence"


def test_decide_after_rerank_low_confidence_is_gate() -> None:
    route, reason = decide_after_rerank(
        error=None,
        reranked_chunks=[_ranked(0.1)],
        confidence_score=0.1,
        threshold=0.25,
    )
    assert route == "gate"
    assert reason == "low_confidence"


def test_decide_after_rerank_empty_is_gate() -> None:
    route, reason = decide_after_rerank(
        error=None,
        reranked_chunks=[],
        confidence_score=0.0,
        threshold=0.25,
    )
    assert route == "gate"
    assert reason == "no_rerank_results"


def test_decide_after_retrieval_strong_to_rerank() -> None:
    route, reason = decide_after_retrieval(
        error=None,
        graded_chunks=[_graded(0.5)],
        ensemble_threshold=0.4,
    )
    assert route == "rerank"
    assert reason is None


def test_decide_after_retrieval_weak_to_gate() -> None:
    route, reason = decide_after_retrieval(
        error=None,
        graded_chunks=[_graded(0.2)],
        ensemble_threshold=0.4,
    )
    assert route == "gate"
    assert reason == "low_retrieval_scores"


def test_retrieval_signal() -> None:
    assert retrieval_signal([]) == 0.0
    assert retrieval_signal([_graded(0.3), _graded(0.7)]) == 0.7


def test_parse_retrieval_gate_dict_continue() -> None:
    target, q = parse_retrieval_gate_decision({"action": "continue"})
    assert target == "reranker"
    assert q is None


def test_parse_rerank_gate_dict_answer() -> None:
    target, q = parse_rerank_gate_decision({"action": "answer"})
    assert target == "answer"
    assert q is None


def test_route_after_qa_policy_rerank_escalate() -> None:
    s = GraphState(qa_after_rerank="escalate")
    assert route_after_qa_policy_rerank(s) == "escalate"


def test_route_after_qa_policy_rerank_answer() -> None:
    s = GraphState(qa_after_rerank="answer")
    assert route_after_qa_policy_rerank(s) == "answer"


def test_route_after_qa_policy_rerank_gate() -> None:
    s = GraphState(qa_after_rerank="gate")
    assert route_after_qa_policy_rerank(s) == "gate"


def test_route_after_qa_policy_rerank_none_defaults_to_escalate() -> None:
    s = GraphState(qa_after_rerank=None)
    assert route_after_qa_policy_rerank(s) == "escalate"


def test_route_after_qa_policy_retrieval() -> None:
    assert (
        route_after_qa_policy_retrieval(GraphState(qa_after_retrieval="rerank"))
        == "rerank"
    )
    assert (
        route_after_qa_policy_retrieval(GraphState(qa_after_retrieval="gate")) == "gate"
    )
    assert (
        route_after_qa_policy_retrieval(GraphState(qa_after_retrieval=None))
        == "escalate"
    )


def test_route_after_qa_retrieval_gate() -> None:
    assert (
        route_after_qa_retrieval_gate(GraphState(qa_retrieval_gate_action="retriever"))
        == "retriever"
    )
    assert (
        route_after_qa_retrieval_gate(GraphState(qa_retrieval_gate_action="reranker"))
        == "reranker"
    )
    assert (
        route_after_qa_retrieval_gate(GraphState(qa_retrieval_gate_action=None))
        == "escalate"
    )


def test_route_after_qa_rerank_gate() -> None:
    assert (
        route_after_qa_rerank_gate(GraphState(qa_rerank_gate_action="retriever"))
        == "retriever"
    )
    assert (
        route_after_qa_rerank_gate(GraphState(qa_rerank_gate_action="answer"))
        == "answer"
    )
    assert (
        route_after_qa_rerank_gate(GraphState(qa_rerank_gate_action=None)) == "escalate"
    )
