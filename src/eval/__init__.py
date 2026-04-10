"""Shared evaluation framework — base protocols, LLM-as-judge, golden datasets.

Agent-specific eval (e.g. librarian RAG eval) extends these base classes.
Dependency direction: agents.*.eval → eval → core (never reversed).
"""
