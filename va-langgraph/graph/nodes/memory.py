"""Memory node — handles 'remember that...' / 'forget...' intents.

Uses a small LLM call to extract the key/value from the user's message,
then persists the preference to the memory store.
"""

from __future__ import annotations

import json
import logging

import shared.memory as memory_store
from shared.model_factory import resolve_chat_model
from shared.schema import AssistantResponse

from ..state import AgentState

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM = """Extract the user's preference action.
Respond with JSON only — no markdown, no explanation.
Format: {"action": "remember" or "forget", "key": "<short snake_case key>", "value": "<value if remembering, else empty>"}

Examples:
  "remember my default currency is EUR"    → {"action": "remember", "key": "currency",    "value": "EUR"}
  "remember I prefer Danish replies"       → {"action": "remember", "key": "language",    "value": "Danish"}
  "remember my VAT rate is 25%"            → {"action": "remember", "key": "vat_rate",    "value": "25%"}
  "forget my language preference"          → {"action": "forget",   "key": "language",    "value": ""}
  "forget everything you know about me"   → {"action": "forget",   "key": "all",         "value": ""}
"""


async def memory_node(state: AgentState) -> AgentState:
    user_id = state.get("user_id", "default")
    messages = state.get("messages", [])
    user_text = str(messages[-1].content) if messages else ""

    try:
        resp = await resolve_chat_model("small").ainvoke([
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": user_text},
        ])
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].strip().lstrip("json").strip()
        parsed = json.loads(raw)
        action = parsed.get("action", "remember")
        key = parsed.get("key", "preference")
        value = parsed.get("value", "")

        if action == "forget" and key == "all":
            # Clear all preferences for this user via repeated deletes is slow;
            # use a direct approach via the sync helper
            import asyncio
            import sqlite3
            import os
            db_path = os.getenv("MEMORY_DB_PATH", "memory.db")

            def _clear_all():
                with sqlite3.connect(db_path) as db:
                    db.execute(
                        "DELETE FROM preference_store WHERE user_id = ? AND key NOT LIKE 'session:%'",
                        (user_id,),
                    )
                    db.commit()

            await asyncio.to_thread(_clear_all)
            msg = "Done — I've forgotten all your preferences."
        elif action == "forget":
            await memory_store.delete(user_id, f"pref:{key}")
            msg = f"Done — I've forgotten your **{key}** preference."
        else:
            await memory_store.upsert(user_id, f"pref:{key}", value)
            msg = f"Got it — I'll remember that **{key}** is **{value}**."

    except Exception as e:
        logger.warning("memory_node failed: %s", e)
        msg = "I had trouble saving that preference. Please try again."

    return {
        **state,
        "response": AssistantResponse(
            message=msg,
            suggestions=["What have you remembered about me?", "Forget all my preferences"],
        ).model_dump(),
    }
