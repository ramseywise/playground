"""Memory load node — reads user preferences at the start of every turn."""

from __future__ import annotations

import memory as memory_store

from ..state import AgentState


async def memory_load_node(state: AgentState) -> AgentState:
    user_id = state.get("user_id", "default")
    prefs = await memory_store.get_top(user_id)
    return {**state, "user_preferences": prefs}
