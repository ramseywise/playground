"""LLM-as-judge graders: registry, presets, composite multi-metric judge, conciseness.

Prompts live next to :data:`METRICS` and pass predicates — single source of truth.

Library-backed RAG metrics stay in :mod:`evals.graders.deepeval` and
:mod:`evals.graders.ragas` (optional deps); re-exported from :mod:`evals.graders`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, ClassVar

from evals.utils.models import EvalTask, GraderKind, GraderResult
from evals.utils.json import strip_json_fences

# ---------------------------------------------------------------------------
# Metric registry (prompts + pass rules)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDefinition:
    """Definition of a single LLM evaluation metric."""

    name: str
    grader_type: str
    standalone_prompt: str
    composite_prompt_section: str
    required_fields: frozenset[str]
    passes: Callable[[dict[str, Any]], bool]


# --- Grounding ---

_GROUNDING_STANDALONE = """\
You are a claim-level grounding verifier for a retrieval-augmented generation system.

Your task is to verify each factual claim in the agent's response against the retrieved \
context passages.  Classify every claim into exactly ONE of four categories:

- **grounded**: directly supported by text in the context.
- **hallucinated**: directly contradicts the context.
- **unverifiable**: neither supported nor contradicted by the context.
- **parametric**: contains specific facts, figures, or claims NOT found in the context \
that appear to come from the model's pre-trained knowledge rather than the retrieved passages.

A claim is parametric (not unverifiable) when it adds information that could only come \
from training data.  Common-sense connectives or linguistic scaffolding do not count as \
parametric.

Steps:
1. Extract up to 7 distinct factual claims from the response.
2. For each claim, classify it as: grounded | hallucinated | unverifiable | parametric.
3. Compute grounding_ratio = claims_grounded / claims_made.
4. Compute parametric_override = parametric_claims / claims_made.

Return ONLY a JSON object with these exact keys:
{
  "claims_made": <integer, number of claims extracted>,
  "claims_grounded": <integer>,
  "claims_hallucinated": <integer>,
  "claims_unverifiable": <integer>,
  "claims_parametric": <integer>,
  "grounding_ratio": <float 0.0-1.0, claims_grounded / claims_made>,
  "has_hallucination": <1.0 if any claim is hallucinated, else 0.0>,
  "parametric_override": <float 0.0-1.0, claims_parametric / claims_made>,
  "is_correct": <true if grounding_ratio >= 0.8 AND has_hallucination == 0.0 AND parametric_override <= 0.2>,
  "score": <same as grounding_ratio>,
  "reasoning": <one sentence citing the most significant ungrounded, hallucinated, or parametric claim>
}
No other text outside the JSON object."""

_GROUNDING_COMPOSITE = """\
## Grounding

Verify each factual claim in the agent's response against the retrieved context passages. \
Classify every claim as: grounded | hallucinated | unverifiable | parametric.

Return the result as a JSON object under the key "grounding":
{
  "claims_made": <integer>,
  "claims_grounded": <integer>,
  "claims_hallucinated": <integer>,
  "claims_unverifiable": <integer>,
  "claims_parametric": <integer>,
  "grounding_ratio": <float 0.0-1.0>,
  "has_hallucination": <1.0 if any claim is hallucinated, else 0.0>,
  "parametric_override": <float 0.0-1.0>,
  "is_correct": <true if grounding_ratio >= 0.8 AND has_hallucination == 0.0 AND parametric_override <= 0.2>,
  "score": <same as grounding_ratio>,
  "reasoning": <one sentence>
}"""


def _grounding_passes(d: dict[str, Any]) -> bool:
    return (
        d.get("grounding_ratio", 0) >= 0.8
        and d.get("has_hallucination", 1) == 0.0
        and d.get("parametric_override", 1) <= 0.2
    )


# --- Completeness ---

_COMPLETENESS_STANDALONE = """\
You are a completeness evaluator for a question-answering system.

Your task is to determine whether the provided answer fully addresses ALL parts of a \
multi-part question. Some questions contain two or more distinct sub-questions; the \
answer must cover each one.

Steps:
1. Identify every distinct sub-question or information need within the user question.
2. For each sub-question, determine whether the answer addresses it (yes/no).
3. For each addressed sub-question, rate the depth of the answer (adequate/superficial).

Return ONLY a JSON object with these exact keys:
{
  "sub_questions_identified": <integer count of distinct sub-questions>,
  "sub_questions_answered": <integer count of sub-questions addressed in the answer>,
  "sub_question_coverage": <float 0.0-1.0, fraction answered>,
  "depth_adequacy": <float 0.0-1.0, average depth of answered sub-questions>,
  "overall_completeness": <float 0.0-1.0, weighted composite>,
  "is_correct": <true if overall_completeness >= 0.7, else false>,
  "score": <same value as overall_completeness>,
  "reasoning": <one sentence explaining which sub-questions were missed or were shallow>
}
No other text outside the JSON object."""

_COMPLETENESS_COMPOSITE = """\
## Completeness

Determine whether the answer fully addresses ALL parts of the question. \
Identify sub-questions, check coverage, and rate depth.

Return the result as a JSON object under the key "completeness":
{
  "sub_questions_identified": <integer>,
  "sub_questions_answered": <integer>,
  "sub_question_coverage": <float 0.0-1.0>,
  "depth_adequacy": <float 0.0-1.0>,
  "overall_completeness": <float 0.0-1.0>,
  "is_correct": <true if overall_completeness >= 0.7>,
  "score": <same as overall_completeness>,
  "reasoning": <one sentence>
}"""


def _completeness_passes(d: dict[str, Any]) -> bool:
    return d.get("overall_completeness", 0) >= 0.7


# --- EPA ---

_EPA_STANDALONE = """\
You are a communication quality evaluator for a customer-facing support AI.

Evaluate the agent's response on three dimensions:

EMPATHY (E): Does the response acknowledge the user's emotional state or difficulty? \
Does the tone feel warm and human? Does it avoid being robotic or dismissive?
0.0 = cold/dismissive, 1.0 = highly empathetic and validating.

PROFESSIONALISM (P): Is the language register appropriate for customer support? \
Is it free from slang, inappropriate content, excessive hedging, or harmful language? \
Does it reflect well on the brand?
0.0 = unprofessional or harmful, 1.0 = polished and brand-appropriate.

ACTIONABILITY (A): Does the response give the user a concrete, actionable next step? \
Is it clear what the user should do to resolve their issue?
0.0 = vague or unhelpful, 1.0 = clear actionable guidance with steps.

Return ONLY a JSON object with these exact keys:
{
  "empathy": <float 0.0-1.0>,
  "professionalism": <float 0.0-1.0>,
  "actionability": <float 0.0-1.0>,
  "epa_composite": <float 0.0-1.0, unweighted mean of the three>,
  "is_correct": <true if epa_composite >= 0.65, else false>,
  "score": <same as epa_composite>,
  "reasoning": <one sentence identifying the weakest dimension and why>
}
No other text outside the JSON object."""

_EPA_COMPOSITE = """\
## EPA (Empathy, Professionalism, Actionability)

Evaluate the agent's response on three communication dimensions: \
Empathy (E), Professionalism (P), and Actionability (A). \
Each scored 0.0-1.0, composite is their unweighted mean.

Return the result as a JSON object under the key "epa":
{
  "empathy": <float 0.0-1.0>,
  "professionalism": <float 0.0-1.0>,
  "actionability": <float 0.0-1.0>,
  "epa_composite": <float 0.0-1.0>,
  "is_correct": <true if epa_composite >= 0.65>,
  "score": <same as epa_composite>,
  "reasoning": <one sentence>
}"""


def _epa_passes(d: dict[str, Any]) -> bool:
    return d.get("epa_composite", 0) >= 0.65


# --- Escalation ---

_ESCALATION_STANDALONE = """\
You are an escalation-appropriateness evaluator for a customer support AI.

Escalation is appropriate when:
- The user's request falls outside the system's documented scope.
- The user has expressed strong negative emotion or urgency requiring human empathy.
- The query requires a human decision (e.g. billing disputes, legal matters, complaints).
- The system could not retrieve any relevant context.

Escalation is inappropriate when:
- The query is within scope and context was available.
- Escalating would leave a resolvable question unanswered.

Steps:
1. Decide whether escalation was WARRANTED by the query and context.
2. Decide whether the agent ACTUALLY escalated (look for phrases like \
"I'll connect you", "let me transfer", "I recommend speaking with a \
representative", or explicit hand-off language).
3. Score appropriateness as 1.0 if warranted and executed are consistent, else 0.0.

Return ONLY a JSON object with these exact keys:
{
  "escalation_warranted": <1.0 if warranted, 0.0 if not>,
  "escalation_executed": <1.0 if the agent did escalate, 0.0 if not>,
  "appropriateness": <1.0 if warranted and executed agree, 0.0 otherwise>,
  "is_correct": <true if appropriateness == 1.0, else false>,
  "score": <same value as appropriateness>,
  "reasoning": <one sentence explaining the decision>
}
No other text outside the JSON object."""

_ESCALATION_COMPOSITE = """\
## Escalation

Evaluate whether the agent correctly escalated or refrained from escalating. \
Escalation is appropriate when the query is out of scope, user is distressed, \
or context is missing. Score 1.0 if warranted and executed agree, else 0.0.

Return the result as a JSON object under the key "escalation":
{
  "escalation_warranted": <1.0 or 0.0>,
  "escalation_executed": <1.0 or 0.0>,
  "appropriateness": <1.0 or 0.0>,
  "is_correct": <true if appropriateness == 1.0>,
  "score": <same as appropriateness>,
  "reasoning": <one sentence>
}"""


def _escalation_passes(d: dict[str, Any]) -> bool:
    return d.get("appropriateness", 0) == 1.0


# --- Friction ---

_FRICTION_STANDALONE = """\
You are a user-friction evaluator for a customer-facing support AI.

High friction means the response makes the user's job harder: unnecessary steps, \
jargon, evasion, asking for information the user already gave, burying the answer, \
or shifting work to the user when a direct answer was possible.

Low friction means: direct, concise, easy to follow, and respectful of the user's time.

Return ONLY a JSON object with these exact keys:
{
  "friction_score": <float 0.0-1.0 where 0.0 = low friction (good), 1.0 = high friction (bad)>,
  "is_correct": <true if friction_score <= 0.35, else false>,
  "score": <float 0.0-1.0, higher is better; use (1.0 - friction_score)>,
  "reasoning": <one sentence citing the main friction source or why friction is low>
}
No other text outside the JSON object."""

_FRICTION_COMPOSITE = """\
## Friction

Rate user friction in the response (0 = low friction/good, 1 = high friction/bad). \
Score should reflect unnecessary burden, evasion, or poor clarity.

Return the result as a JSON object under the key "friction":
{
  "friction_score": <float 0.0-1.0>,
  "is_correct": <true if friction_score <= 0.35>,
  "score": <1.0 - friction_score, higher is better>,
  "reasoning": <one sentence>
}"""


def _friction_passes(d: dict[str, Any]) -> bool:
    return float(d.get("friction_score", 1.0)) <= 0.35


METRICS: dict[str, MetricDefinition] = {
    "grounding": MetricDefinition(
        name="grounding",
        grader_type="grounding",
        standalone_prompt=_GROUNDING_STANDALONE,
        composite_prompt_section=_GROUNDING_COMPOSITE,
        required_fields=frozenset({"context", "response", "query"}),
        passes=_grounding_passes,
    ),
    "completeness": MetricDefinition(
        name="completeness",
        grader_type="completeness_judge",
        standalone_prompt=_COMPLETENESS_STANDALONE,
        composite_prompt_section=_COMPLETENESS_COMPOSITE,
        required_fields=frozenset({"query", "response", "expected"}),
        passes=_completeness_passes,
    ),
    "epa": MetricDefinition(
        name="epa",
        grader_type="epa_judge",
        standalone_prompt=_EPA_STANDALONE,
        composite_prompt_section=_EPA_COMPOSITE,
        required_fields=frozenset({"query", "response"}),
        passes=_epa_passes,
    ),
    "escalation": MetricDefinition(
        name="escalation",
        grader_type="escalation_judge",
        standalone_prompt=_ESCALATION_STANDALONE,
        composite_prompt_section=_ESCALATION_COMPOSITE,
        required_fields=frozenset({"query", "context", "response"}),
        passes=_escalation_passes,
    ),
    "friction": MetricDefinition(
        name="friction",
        grader_type="friction_judge",
        standalone_prompt=_FRICTION_STANDALONE,
        composite_prompt_section=_FRICTION_COMPOSITE,
        required_fields=frozenset({"query", "response"}),
        passes=_friction_passes,
    ),
}

_MG = METRICS["grounding"]
_MC = METRICS["completeness"]
_ME = METRICS["epa"]
_MS = METRICS["escalation"]
_MF = METRICS["friction"]

# ---------------------------------------------------------------------------
# Base + preset judges
# ---------------------------------------------------------------------------


class LLMJudge:
    """Base LLM-as-judge: format prompt -> call LLM -> parse JSON verdict."""

    system_prompt: str = "You are an evaluation judge."
    grader_type: str = "llm_judge"
    grader_kind: ClassVar[GraderKind] = GraderKind.LLM_JUDGE

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def grade(self, task: EvalTask) -> GraderResult:
        response = task.metadata.get("response", "")
        user_msg = self._format_user_message(
            query=task.query,
            response=response,
            context=task.context,
            expected=task.expected_answer,
        )
        raw = await self._llm.generate(
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=1024,
        )
        return self._parse_result(raw, task_id=task.id)

    def _format_user_message(
        self,
        *,
        query: str,
        response: str,
        context: str,
        expected: str,
    ) -> str:
        parts = [f"Question: {query}", f"Answer: {response}"]
        if context:
            parts.insert(1, f"Context: {context}")
        if expected:
            parts.append(f"Expected: {expected}")
        return "\n\n".join(parts)

    def _parse_result(self, raw: str, task_id: str = "") -> GraderResult:
        try:
            data: dict[str, Any] = json.loads(strip_json_fences(raw))
            score = float(data.get("score", 0.0))
            return GraderResult(
                task_id=task_id,
                grader_type=self.grader_type,
                is_correct=bool(data.get("is_correct", False)),
                score=score,
                reasoning=str(data.get("reasoning", "")),
                dimensions={
                    k: float(v)
                    for k, v in data.items()
                    if k not in {"is_correct", "score", "reasoning"}
                    and isinstance(v, (int, float))
                },
                details={
                    k: v
                    for k, v in data.items()
                    if k not in {"is_correct", "score", "reasoning"}
                },
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return GraderResult(
                task_id=task_id,
                grader_type=self.grader_type,
                is_correct=False,
                score=0.0,
                reasoning=f"Failed to parse judge response: {raw[:200]}",
            )


class GroundingJudge(LLMJudge):
    """Claim-level grounding and parametric-knowledge check."""

    system_prompt: str = _MG.standalone_prompt
    grader_type: str = _MG.grader_type

    def _format_user_message(
        self,
        *,
        query: str,
        response: str,
        context: str,
        expected: str,
    ) -> str:
        parts = [
            f"Retrieved context passages:\n{context}"
            if context
            else "Retrieved context passages: [none]",
            f"Agent response:\n{response}",
            f"User query (for reference): {query}",
        ]
        return "\n\n".join(parts)


class CompletenessJudge(LLMJudge):
    """Whether a multi-part question is fully answered."""

    system_prompt: str = _MC.standalone_prompt
    grader_type: str = _MC.grader_type

    def _format_user_message(
        self,
        *,
        query: str,
        response: str,
        context: str,
        expected: str,
    ) -> str:
        parts = [f"Question: {query}", f"Answer: {response}"]
        if expected:
            parts.append(f"Expected answer (reference): {expected}")
        return "\n\n".join(parts)


class EPAJudge(LLMJudge):
    """Empathy, professionalism, actionability."""

    system_prompt: str = _ME.standalone_prompt
    grader_type: str = _ME.grader_type

    def _format_user_message(
        self,
        *,
        query: str,
        response: str,
        context: str,
        expected: str,
    ) -> str:
        return f"User query: {query}\n\nAgent response: {response}"


class EscalationJudge(LLMJudge):
    """Whether escalation behaviour matched the situation."""

    system_prompt: str = _MS.standalone_prompt
    grader_type: str = _MS.grader_type

    def _format_user_message(
        self,
        *,
        query: str,
        response: str,
        context: str,
        expected: str,
    ) -> str:
        parts = [
            f"Query: {query}",
            f"Context available to agent: {context}"
            if context
            else "Context available to agent: [none]",
            f"Agent response: {response}",
        ]
        return "\n\n".join(parts)


class FrictionJudge(LLMJudge):
    """Low UX friction: direct, concise, no runaround."""

    system_prompt: str = _MF.standalone_prompt
    grader_type: str = _MF.grader_type


# ---------------------------------------------------------------------------
# Composite judge
# ---------------------------------------------------------------------------

_COMPOSITE_PREAMBLE = """\
You are a multi-metric evaluation judge. Evaluate the provided input on each \
of the following dimensions. For EACH dimension return a JSON sub-object under \
its designated key. Return ONLY a single JSON object containing all the \
per-metric sub-objects — no other text.
"""

_COMPOSITE_FOOTER = """
Return ONLY a JSON object with one top-level key per metric, exactly as named above. \
No other text outside the JSON object."""


class CompositeJudge:
    """Multi-metric LLM judge — evaluates selected metrics in one call."""

    grader_kind: ClassVar[GraderKind] = GraderKind.LLM_JUDGE

    def __init__(self, llm: Any, metrics: list[str]) -> None:
        unknown = set(metrics) - set(METRICS)
        if unknown:
            msg = f"Unknown metrics: {sorted(unknown)}. Available: {sorted(METRICS)}"
            raise ValueError(msg)
        if not metrics:
            msg = "At least one metric must be selected"
            raise ValueError(msg)

        self._llm = llm
        self._metric_names = list(metrics)
        self._metric_defs: list[MetricDefinition] = [METRICS[m] for m in metrics]
        self._system_prompt = self._build_system_prompt()

    @property
    def grader_type(self) -> str:
        return "composite:" + "+".join(sorted(self._metric_names))

    async def grade(self, task: EvalTask) -> GraderResult:
        response = task.metadata.get("response", "")
        user_msg = self._format_user_message(
            query=task.query,
            response=response,
            context=task.context,
            expected=task.expected_answer,
        )
        raw = await self._llm.generate(
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=2048,
        )
        return self._parse_combined_result(raw, task_id=task.id)

    def _build_system_prompt(self) -> str:
        sections = [_COMPOSITE_PREAMBLE]
        for mdef in self._metric_defs:
            sections.append(mdef.composite_prompt_section)
        sections.append(_COMPOSITE_FOOTER)
        return "\n".join(sections)

    def _format_user_message(
        self,
        *,
        query: str,
        response: str,
        context: str,
        expected: str,
    ) -> str:
        needed = frozenset().union(*(m.required_fields for m in self._metric_defs))

        parts: list[str] = []
        if "context" in needed:
            parts.append(
                f"Retrieved context passages:\n{context}"
                if context
                else "Retrieved context passages: [none]"
            )
        parts.append(f"Agent response:\n{response}")
        if "query" in needed:
            parts.append(f"User query: {query}")
        if "expected" in needed and expected:
            parts.append(f"Expected answer (reference): {expected}")
        return "\n\n".join(parts)

    def _parse_combined_result(self, raw: str, task_id: str) -> GraderResult:
        try:
            data: dict[str, Any] = json.loads(strip_json_fences(raw))
        except (json.JSONDecodeError, ValueError):
            return GraderResult(
                task_id=task_id,
                grader_type=self.grader_type,
                is_correct=False,
                score=0.0,
                reasoning=f"Failed to parse combined judge response: {raw[:200]}",
            )

        all_correct: list[bool] = []
        all_scores: list[float] = []
        all_reasonings: list[str] = []
        dimensions: dict[str, float] = {}
        details: dict[str, Any] = {}

        for mdef in self._metric_defs:
            sub = data.get(mdef.name, {})
            if not isinstance(sub, dict):
                sub = {}

            correct = mdef.passes(sub)
            score = float(sub.get("score", 0.0))
            reasoning = str(sub.get("reasoning", ""))

            all_correct.append(correct)
            all_scores.append(score)
            if reasoning:
                all_reasonings.append(f"[{mdef.name}] {reasoning}")

            for k, v in sub.items():
                if k not in {"is_correct", "score", "reasoning"} and isinstance(
                    v, (int, float)
                ):
                    dimensions[f"{mdef.name}.{k}"] = float(v)
                if k not in {"score", "reasoning"}:
                    details[f"{mdef.name}.{k}"] = v

        return GraderResult(
            task_id=task_id,
            grader_type=self.grader_type,
            is_correct=all(all_correct),
            score=sum(all_scores) / len(all_scores) if all_scores else 0.0,
            reasoning="; ".join(all_reasonings),
            dimensions=dimensions,
            details=details,
        )


# ---------------------------------------------------------------------------
# Conciseness (hybrid)
# ---------------------------------------------------------------------------

_PADDING_SYSTEM = """\
You are a conciseness evaluator for a customer support AI.

Assess whether the response is appropriately concise or contains unnecessary padding. \
Padding includes: excessive preamble, repetition of the question, hedging disclaimers \
that add no value, and filler phrases like "Certainly!", "Great question!", "Of course!".

Return ONLY a JSON object with these exact keys:
{
  "padding_score": <float 0.0-1.0, where 1.0 means no padding and 0.0 means severe padding>,
  "is_correct": <true if padding_score >= 0.7, else false>,
  "score": <same as padding_score>,
  "reasoning": <one sentence identifying the main source of padding if any>
}
No other text outside the JSON object."""


class ConcisenessGrader:
    """Hybrid conciseness grader: token budget + optional LLM padding check."""

    grader_type: str = "conciseness"
    grader_kind: ClassVar[GraderKind] = GraderKind.LLM_JUDGE

    def __init__(
        self,
        llm: Any | None = None,
        *,
        max_ratio: float = 2.0,
        expected_tokens: int = 150,
    ) -> None:
        self._llm = llm
        self._max_ratio = max_ratio
        self._expected_tokens = expected_tokens

    async def grade(self, task: EvalTask) -> GraderResult:
        response = task.metadata.get("response", "")
        response_tokens = len(response.split())

        token_ratio = response_tokens / max(self._expected_tokens, 1)
        within_budget = 1.0 if token_ratio <= self._max_ratio else 0.0
        score = min(1.0, 1.0 / max(token_ratio, 0.01))

        dimensions: dict[str, float] = {
            "token_ratio": round(token_ratio, 4),
            "within_budget": within_budget,
        }

        if self._llm is not None:
            padding_score = await self._evaluate_padding(response)
            dimensions["padding_score"] = padding_score

        is_correct = within_budget == 1.0
        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            is_correct=is_correct,
            score=score,
            reasoning=f"token_ratio={token_ratio:.2f}, within_budget={within_budget == 1.0}",
            dimensions=dimensions,
        )

    async def _evaluate_padding(self, response: str) -> float:
        assert self._llm is not None  # noqa: S101
        raw = await self._llm.generate(
            system=_PADDING_SYSTEM,
            messages=[{"role": "user", "content": f"Agent response:\n{response}"}],
            max_tokens=512,
        )
        try:
            data: dict[str, Any] = json.loads(strip_json_fences(raw))
            return float(data.get("padding_score", 0.0))
        except (json.JSONDecodeError, KeyError, ValueError):
            return 0.0


__all__ = [
    "CompletenessJudge",
    "CompositeJudge",
    "ConcisenessGrader",
    "EPAJudge",
    "EscalationJudge",
    "FrictionJudge",
    "GroundingJudge",
    "LLMJudge",
    "METRICS",
    "MetricDefinition",
]
