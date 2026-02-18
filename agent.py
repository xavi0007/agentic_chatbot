from __future__ import annotations

from dataclasses import dataclass

from .planner import Planner
from .schemas import Action, AgentResponse, ChatMessage
from .skills import ClarifySkill, JokeSkill, RecipeSkill


@dataclass
class AgenticChatbot:
    planner: Planner
    clarify_skill: ClarifySkill
    joke_skill: JokeSkill
    recipe_skill: RecipeSkill

    def respond(self, history: list[ChatMessage], user_message: str) -> AgentResponse:
        plan = self.planner.plan(history=history, user_message=user_message)

        if plan.action == Action.JOKE:
            return self.joke_skill.run(plan, history, user_message)
        if plan.action == Action.RECIPE:
            return self.recipe_skill.run(plan, history, user_message)
        return self.clarify_skill.run(plan, history, user_message)
