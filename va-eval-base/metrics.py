"""Service-specific metric layers for RAG and orchestration services."""

from __future__ import annotations

from .models import EvalTask, GraderResult, ServiceResponse


class RAGMetricsGrader:
    """Compute retrieval-specific metrics for va-support-rag."""

    grader_type = "rag_metrics"

    async def grade(self, task: EvalTask, response: ServiceResponse) -> GraderResult:
        """Score RAG performance: retrieval quality, citations, escalation."""
        if response.service != "va-support-rag":
            return GraderResult(
                task_id=task.id,
                grader_type=self.grader_type,
                service=response.service,
                is_correct=False,
                score=0.0,
                reasoning="Not a RAG service",
            )

        metadata = response.metadata or {}
        citations = metadata.get("citations", [])
        escalated = metadata.get("escalated", False)
        pipeline_error = metadata.get("pipeline_error", False)

        # Score factors:
        # - Has citations (retrieval found relevant docs)
        # - Not escalated (confident answer)
        # - No pipeline errors
        has_citations = len(citations) > 0
        not_escalated = not escalated
        no_error = not pipeline_error

        score = (
            float(has_citations) * 0.4
            + float(not_escalated) * 0.4
            + float(no_error) * 0.2
        )

        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            service=response.service,
            is_correct=score >= 0.7,
            score=score,
            reasoning=(
                f"citations={len(citations)}, "
                f"escalated={escalated}, error={pipeline_error}"
            ),
            dimensions={
                "has_citations": float(has_citations),
                "not_escalated": float(not_escalated),
                "no_error": float(no_error),
            },
            details=metadata,
        )


class OrchestrationMetricsGrader:
    """Compute orchestration-specific metrics for va-google-adk and va-langgraph."""

    grader_type = "orchestration_metrics"

    async def grade(self, task: EvalTask, response: ServiceResponse) -> GraderResult:
        """Score orchestration performance: routing, suggestions, navigation."""
        if response.service == "va-support-rag":
            return GraderResult(
                task_id=task.id,
                grader_type=self.grader_type,
                service=response.service,
                is_correct=False,
                score=0.0,
                reasoning="Not an orchestration service",
            )

        # Score factors:
        # - Routed to correct domain (if expected_intent is set)
        # - Generated suggestions (helpfulness signal)
        # - No errors
        has_suggestions = len(response.suggestions) > 0
        no_error = not response.error
        routed_correctly = (
            response.classified_intent == task.expected_intent
            if task.expected_intent
            else True
        )

        score = (
            float(routed_correctly) * 0.5
            + float(has_suggestions) * 0.3
            + float(no_error) * 0.2
        )

        return GraderResult(
            task_id=task.id,
            grader_type=self.grader_type,
            service=response.service,
            is_correct=score >= 0.7,
            score=score,
            reasoning=(
                f"routing={routed_correctly}, "
                f"suggestions={len(response.suggestions)}, error={response.error}"
            ),
            dimensions={
                "routed_correctly": float(routed_correctly),
                "has_suggestions": float(has_suggestions),
                "no_error": float(no_error),
            },
        )
