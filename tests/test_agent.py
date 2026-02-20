from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentic_chatbot.agent import AgenticChatbot
from agentic_chatbot.schemas import Action, AgentResponse, Plan


@dataclass
class StubPlanner:
    next_plan: Plan

    def plan(self, history, user_message):
        return self.next_plan


@dataclass
class StubSkill:
    response: AgentResponse
    call_count: int = 0

    def run(self, plan, history, user_message):
        self.call_count += 1
        return self.response


class AgentTests(unittest.TestCase):
    def test_agent_routes_to_joke_skill(self) -> None:
        planner = StubPlanner(next_plan=Plan(action=Action.JOKE, reason="joke"))
        clarify = StubSkill(response=AgentResponse(content="clarify", action=Action.CLARIFY))
        joke = StubSkill(response=AgentResponse(content="joke", action=Action.JOKE))
        recipe = StubSkill(response=AgentResponse(content="recipe", action=Action.RECIPE))
        agent = AgenticChatbot(planner=planner, clarify_skill=clarify, joke_skill=joke, recipe_skill=recipe)

        response = agent.respond(history=[], user_message="tell joke")

        self.assertEqual(response.action, Action.JOKE)
        self.assertEqual(joke.call_count, 1)
        self.assertEqual(recipe.call_count, 0)
        self.assertEqual(clarify.call_count, 0)

    def test_agent_routes_to_recipe_skill(self) -> None:
        planner = StubPlanner(next_plan=Plan(action=Action.RECIPE, reason="recipe"))
        clarify = StubSkill(response=AgentResponse(content="clarify", action=Action.CLARIFY))
        joke = StubSkill(response=AgentResponse(content="joke", action=Action.JOKE))
        recipe = StubSkill(response=AgentResponse(content="recipe", action=Action.RECIPE))
        agent = AgenticChatbot(planner=planner, clarify_skill=clarify, joke_skill=joke, recipe_skill=recipe)

        response = agent.respond(history=[], user_message="recipe")

        self.assertEqual(response.action, Action.RECIPE)
        self.assertEqual(joke.call_count, 0)
        self.assertEqual(recipe.call_count, 1)
        self.assertEqual(clarify.call_count, 0)

    def test_agent_routes_to_clarify_for_non_joke_recipe(self) -> None:
        planner = StubPlanner(next_plan=Plan(action=Action.CLARIFY, reason="ambiguous"))
        clarify = StubSkill(response=AgentResponse(content="clarify", action=Action.CLARIFY))
        joke = StubSkill(response=AgentResponse(content="joke", action=Action.JOKE))
        recipe = StubSkill(response=AgentResponse(content="recipe", action=Action.RECIPE))
        agent = AgenticChatbot(planner=planner, clarify_skill=clarify, joke_skill=joke, recipe_skill=recipe)

        response = agent.respond(history=[], user_message="help")

        self.assertEqual(response.action, Action.CLARIFY)
        self.assertEqual(joke.call_count, 0)
        self.assertEqual(recipe.call_count, 0)
        self.assertEqual(clarify.call_count, 1)


if __name__ == "__main__":
    unittest.main()
