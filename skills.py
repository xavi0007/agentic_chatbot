from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .llm import LLMClient
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

    def run(self, plan: Plan, history: list[ChatMessage], user_message: str) -> AgentResponse:
        topic = str(plan.params.get("topic", "anything")).strip() or "anything"
        style = str(plan.params.get("style", "clean")).strip() or "clean"
        prompt = (
            "Write one short joke. "
            f"Topic: {topic}. Style: {style}. "
            "Keep it safe for work."
        )
        content = self.llm.complete(
            [
                ChatMessage(role=Role.SYSTEM, content="You are a concise comedian."),
                *history,
                ChatMessage(role=Role.USER, content=prompt),
            ],
            temperature=0.8,
        )
        return AgentResponse(content=content.strip(), action=Action.JOKE)


@dataclass
class RecipeSkill:
    llm: LLMClient

    def run(self, plan: Plan, history: list[ChatMessage], user_message: str) -> AgentResponse:
        ingredients = plan.params.get("ingredients", "")
        servings = plan.params.get("servings", 2)
        diet = plan.params.get("diet", "none")

        prompt = (
            "Create a practical recipe with title, ingredients, and steps. "
            f"Ingredients preference: {ingredients}. Servings: {servings}. Diet: {diet}. "
            "Keep it under 12 steps and include estimated total time."
        )
        content = self.llm.complete(
            [
                ChatMessage(role=Role.SYSTEM, content="You are a precise cooking assistant."),
                *history,
                ChatMessage(role=Role.USER, content=prompt),
            ],
            temperature=0.4,
        )
        return AgentResponse(content=content.strip(), action=Action.RECIPE)
