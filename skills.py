from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .llm import LLMClient
from .mcp import MCPConnectorRegistry
from .schemas import Action, AgentResponse, ChatMessage, Plan, Role


class Skill(Protocol):
    def run(self, plan: Plan, history: list[ChatMessage], user_message: str) -> AgentResponse:
        ...


@dataclass
class ClarifySkill:
    default_question: str = "Could you clarify what you want: a joke or a food recipe?"

    def run(self, plan: Plan, history: list[ChatMessage], user_message: str) -> AgentResponse:
        question = plan.clarifying_question or self.default_question
        return AgentResponse(content=question, action=Action.CLARIFY)


@dataclass
class JokeSkill:
    llm: LLMClient
    mcp_registry: MCPConnectorRegistry | None = None

    def run(self, plan: Plan, history: list[ChatMessage], user_message: str) -> AgentResponse:
        topic = str(plan.params.get("topic", "anything")).strip() or "anything"
        style = str(plan.params.get("style", "clean")).strip() or "clean"
        system_prompt = self._resolve_system_prompt(
            default_prompt="You are a concise comedian.",
            plan=plan,
            user_message=user_message,
        )
        tool_context = self._resolve_tool_context(plan=plan, user_message=user_message)
        prompt = (
            "Write one short joke. "
            f"Topic: {topic}. Style: {style}. "
            "Keep it safe for work."
            f"{tool_context}"
        )
        content = self.llm.complete(
            [
                ChatMessage(role=Role.SYSTEM, content=system_prompt),
                *history,
                ChatMessage(role=Role.USER, content=prompt),
            ],
            temperature=0.8,
        )
        return AgentResponse(content=content.strip(), action=Action.JOKE)

    def _resolve_system_prompt(self, *, default_prompt: str, plan: Plan, user_message: str) -> str:
        if not self.mcp_registry:
            return default_prompt
        alias = plan.params.get("mcp_prompt")
        if not isinstance(alias, str) or not alias.strip():
            return default_prompt

        prompt_args = _dict_param(plan.params.get("prompt_args"))
        prompt_args.setdefault("user_message", user_message)
        try:
            resolved = self.mcp_registry.get_prompt(alias.strip(), prompt_args)
            return resolved.strip() or default_prompt
        except (KeyError, ValueError):
            return default_prompt

    def _resolve_tool_context(self, *, plan: Plan, user_message: str) -> str:
        if not self.mcp_registry:
            return ""
        alias = plan.params.get("mcp_tool")
        if not isinstance(alias, str) or not alias.strip():
            return ""

        tool_args = _dict_param(plan.params.get("tool_args"))
        tool_args.setdefault("user_message", user_message)
        try:
            tool_result = self.mcp_registry.call_tool(alias.strip(), tool_args).strip()
            return f"\nExternal tool context:\n{tool_result}" if tool_result else ""
        except (KeyError, ValueError):
            return ""


@dataclass
class RecipeSkill:
    llm: LLMClient
    mcp_registry: MCPConnectorRegistry | None = None

    def run(self, plan: Plan, history: list[ChatMessage], user_message: str) -> AgentResponse:
        ingredients = plan.params.get("ingredients", "")
        servings = plan.params.get("servings", 2)
        diet = plan.params.get("diet", "none")
        system_prompt = self._resolve_system_prompt(
            default_prompt="You are a precise cooking assistant.",
            plan=plan,
            user_message=user_message,
        )
        tool_context = self._resolve_tool_context(plan=plan, user_message=user_message)

        prompt = (
            "Create a practical recipe with title, ingredients, and steps. "
            f"Ingredients preference: {ingredients}. Servings: {servings}. Diet: {diet}. "
            "Keep it under 12 steps and include estimated total time."
            f"{tool_context}"
        )
        content = self.llm.complete(
            [
                ChatMessage(role=Role.SYSTEM, content=system_prompt),
                *history,
                ChatMessage(role=Role.USER, content=prompt),
            ],
            temperature=0.4,
        )
        return AgentResponse(content=content.strip(), action=Action.RECIPE)

    def _resolve_system_prompt(self, *, default_prompt: str, plan: Plan, user_message: str) -> str:
        if not self.mcp_registry:
            return default_prompt
        alias = plan.params.get("mcp_prompt")
        if not isinstance(alias, str) or not alias.strip():
            return default_prompt

        prompt_args = _dict_param(plan.params.get("prompt_args"))
        prompt_args.setdefault("user_message", user_message)
        try:
            resolved = self.mcp_registry.get_prompt(alias.strip(), prompt_args)
            return resolved.strip() or default_prompt
        except (KeyError, ValueError):
            return default_prompt

    def _resolve_tool_context(self, *, plan: Plan, user_message: str) -> str:
        if not self.mcp_registry:
            return ""
        alias = plan.params.get("mcp_tool")
        if not isinstance(alias, str) or not alias.strip():
            return ""

        tool_args = _dict_param(plan.params.get("tool_args"))
        tool_args.setdefault("user_message", user_message)
        try:
            tool_result = self.mcp_registry.call_tool(alias.strip(), tool_args).strip()
            return f"\nExternal tool context:\n{tool_result}" if tool_result else ""
        except (KeyError, ValueError):
            return ""


def _dict_param(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
