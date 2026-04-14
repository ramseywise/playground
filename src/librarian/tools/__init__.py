"""Framework-agnostic tool abstractions for the Librarian pipeline.

Each tool has explicit Pydantic input/output schemas and a single
``run()`` method.  LangGraph and ADK adapters consume these directly.
"""
