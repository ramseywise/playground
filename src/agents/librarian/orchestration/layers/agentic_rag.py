from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rag_system.src.rag_core.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class RetrievalTool:
    """Tool the agent can use to retrieve more information."""

    name: str = "search_knowledge_base"
    description: str = (
        "Search the knowledge base for information. Use when you need more context."
    )

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "focus": {
                        "type": "string",
                        "enum": ["how_to", "troubleshooting", "factual", "general"],
                        "description": "Type of information to focus on",
                    },
                },
                "required": ["query"],
            },
        }


@dataclass
class AgentState:
    """State for agentic retrieval loop."""

    original_query: str
    current_context: list[dict] = field(default_factory=list)
    retrieval_history: list[dict] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 3
    is_complete: bool = False
    final_answer: str | None = None


class AgenticRAG:
    """RAG pipeline with agentic retrieval — generator can request follow-up retrievals."""

    def __init__(
        self,
        retriever,
        generator,
        query_analyzer=None,
        max_iterations: int = 3,
        max_context_docs: int = 10,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.query_analyzer = query_analyzer
        self.max_iterations = max_iterations
        self.max_context_docs = max_context_docs
        self.tool = RetrievalTool()

    def run(self, query: str, context: dict | None = None) -> dict[str, Any]:
        state = AgentState(original_query=query, max_iterations=self.max_iterations)

        initial_results = self._retrieve(query)
        state.current_context.extend(initial_results)
        state.retrieval_history.append(
            {"iteration": 0, "query": query, "num_results": len(initial_results)}
        )

        while not state.is_complete and state.iteration < state.max_iterations:
            state.iteration += 1
            result = self._generate_with_tools(state, context)

            if result.get("tool_calls"):
                for tool_call in result["tool_calls"]:
                    if tool_call["name"] == self.tool.name:
                        follow_up_query = tool_call["arguments"].get("query", "")
                        follow_up_results = self._retrieve(follow_up_query)
                        self._merge_context(state, follow_up_results)
                        state.retrieval_history.append(
                            {
                                "iteration": state.iteration,
                                "query": follow_up_query,
                                "num_results": len(follow_up_results),
                            }
                        )
            else:
                state.is_complete = True
                state.final_answer = result.get("answer", "")

        if not state.final_answer:
            state.final_answer = self._generate_final(state, context)

        return {
            "answer": state.final_answer,
            "iterations": state.iteration,
            "retrieval_history": state.retrieval_history,
            "total_context_docs": len(state.current_context),
        }

    def _retrieve(self, query: str, k: int = 5) -> list[dict]:
        results = self.retriever.retrieve(query, k=k)
        return [r.document for r in results]

    def _merge_context(self, state: AgentState, new_docs: list[dict]) -> None:
        existing_urls = {
            doc.get("url", doc.get("URL", "")) for doc in state.current_context
        }
        for doc in new_docs:
            url = doc.get("url", doc.get("URL", ""))
            if (
                url not in existing_urls
                and len(state.current_context) < self.max_context_docs
            ):
                state.current_context.append(doc)
                existing_urls.add(url)

    def _generate_with_tools(
        self, state: AgentState, context: dict | None
    ) -> dict[str, Any]:
        tool_prompt = self._build_tool_prompt(state)
        try:
            response = self.generator.generate(
                query=tool_prompt, docs=state.current_context, context=context
            )
            return self._parse_tool_response(response)
        except Exception as exc:
            log.error("agentic_rag.generate_failed", error=str(exc))
            return {"answer": None, "tool_calls": None}

    def _build_tool_prompt(self, state: AgentState) -> str:
        return (
            f"Answer the following question based on the provided documents.\n\n"
            f"If you need more information, request a follow-up search with: [[SEARCH: your query]]\n\n"
            f"Question: {state.original_query}\n\n"
            f"Iteration: {state.iteration}/{state.max_iterations}\n\n"
            f"If you have enough information, answer directly. Otherwise, use [[SEARCH: ...]]."
        )

    def _parse_tool_response(self, response: str) -> dict[str, Any]:
        search_match = re.search(r"\[\[SEARCH:\s*(.+?)\]\]", response)
        if search_match:
            return {
                "answer": None,
                "tool_calls": [
                    {
                        "name": self.tool.name,
                        "arguments": {"query": search_match.group(1).strip()},
                    }
                ],
            }
        return {"answer": response, "tool_calls": None}

    def _generate_final(self, state: AgentState, context: dict | None) -> str:
        return self.generator.generate(
            query=state.original_query, docs=state.current_context, context=context
        )


class DecomposedQueryRAG:
    """RAG that handles complex queries by decomposing into sub-queries."""

    def __init__(
        self, retriever, generator, query_analyzer, max_sub_queries: int = 3
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.query_analyzer = query_analyzer
        self.max_sub_queries = max_sub_queries

    def run(self, query: str, context: dict | None = None) -> dict[str, Any]:
        analysis = self.query_analyzer.analyze(query)
        sub_queries = analysis.sub_queries[: self.max_sub_queries]

        if len(sub_queries) <= 1:
            results = self.retriever.retrieve(query, k=5)
            docs = [r.document for r in results]
            answer = self.generator.generate(query, docs, context)
            return {"answer": answer, "decomposed": False, "sub_queries": []}

        all_docs: list[dict] = []
        sub_results: list[dict] = []

        for sub_q in sub_queries:
            results = self.retriever.retrieve(sub_q, k=3)
            docs = [r.document for r in results]
            all_docs.extend(docs)
            sub_results.append({"query": sub_q, "num_docs": len(docs)})

        seen_urls: set[str] = set()
        unique_docs: list[dict] = []
        for doc in all_docs:
            url = doc.get("url", doc.get("URL", ""))
            if url not in seen_urls:
                seen_urls.add(url)
                unique_docs.append(doc)

        sub_q_list = "\n".join(f"- {sq}" for sq in sub_queries)
        combined_prompt = (
            f"The original question was: {query}\n\n"
            f"This was decomposed into:\n{sub_q_list}\n\n"
            f"Please answer the original question fully based on the documents."
        )

        answer = self.generator.generate(combined_prompt, unique_docs, context)
        return {
            "answer": answer,
            "decomposed": True,
            "sub_queries": sub_results,
            "total_docs": len(unique_docs),
        }
