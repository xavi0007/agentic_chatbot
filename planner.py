from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .llm import LLMClient
from .schemas import Action, ChatMessage, Plan, Role


PLANNER_SYSTEM_PROMPT = """You are a routing planner for a chatbot.
Pick exactly one action from: clarify, joke, recipe.
Return strict JSON with this shape:
{
  \"action\": \"clarify|joke|recipe\",
  \"reason\": \"short reason\",
  \"params\": {\"any\": \"json object\"},
  \"clarifying_question\": \"string or null\"
}
Rules:
- choose clarify if the user request is ambiguous.
- choose joke for joke/comedy requests.
- choose recipe for food/meal/cooking requests.
- do not include markdown or extra prose.
"""


@dataclass
class Planner:
    llm: LLMClient

    def plan(self, history: list[ChatMessage], user_message: str) -> Plan:
        prompt_messages = [
            ChatMessage(role=Role.SYSTEM, content=PLANNER_SYSTEM_PROMPT),
            *history,
            ChatMessage(role=Role.USER, content=user_message),
        ]
        raw = self.llm.complete(prompt_messages, temperature=0)
        data = _safe_parse_json(raw)
        if data is None:
            return Plan(
                action=Action.CLARIFY,
                reason="Planner output was not valid JSON",
                clarifying_question="Could you clarify whether you want a joke or a recipe?",
            )

        action_value = str(data.get("action", "clarify")).strip().lower()
        try:
            action = Action(action_value)
        except ValueError:
            action = Action.CLARIFY

        reason = str(data.get("reason", "No reason provided")).strip() or "No reason provided"
        params = data.get("params", {})
        if not isinstance(params, dict):
            params = {}

        clarifying_question = data.get("clarifying_question")
        if clarifying_question is not None:
            clarifying_question = str(clarifying_question).strip() or None

        return Plan(
            action=action,
            reason=reason,
            params=params,
            clarifying_question=clarifying_question,
        )


def _safe_parse_json(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None
