"""Central registry of LLM evaluation metrics.

Each ``MetricDefinition`` bundles a metric's system prompt, pass criteria,
and required input fields.  Used by both standalone judge classes and the
``CompositeJudge`` for multi-metric evaluation in a single LLM call.

No imports from other ``eval.graders`` submodules — this module is the
dependency root for all LLM judge classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class MetricDefinition:
    """Definition of a single LLM evaluation metric."""

    name: str
    grader_type: str
    standalone_prompt: str
    composite_prompt_section: str
    required_fields: frozenset[str]
    passes: Callable[[dict[str, Any]], bool]


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# EPA (Empathy, Professionalism, Actionability)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

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
}
