from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentic_chatbot.planner import Planner
from agentic_chatbot.schemas import Action, ChatMessage, Role


@dataclass
class FakeLLM:
    output: str
    last_messages: list[ChatMessage] | None = None
    last_temperature: float | None = None

    def complete(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> str:
        self.last_messages = messages
        self.last_temperature = temperature
        return self.output


class PlannerTests(unittest.TestCase):
    def test_plan_parses_valid_json(self) -> None:
        llm = FakeLLM(
            output=(
                '{"action":"joke","reason":"user asked for one",'
                '"params":{"topic":"cats","style":"dry"},'
                '"clarifying_question":null}'
            )
        )
        planner = Planner(llm=llm)

        plan = planner.plan(history=[], user_message="Tell me a joke about cats")

        self.assertEqual(plan.action, Action.JOKE)
        self.assertEqual(plan.reason, "user asked for one")
        self.assertEqual(plan.params, {"topic": "cats", "style": "dry"})
        self.assertIsNone(plan.clarifying_question)
        self.assertEqual(llm.last_temperature, 0)
        self.assertIsNotNone(llm.last_messages)
        assert llm.last_messages is not None
        self.assertEqual(llm.last_messages[0].role, Role.SYSTEM)
        self.assertEqual(llm.last_messages[-1].content, "Tell me a joke about cats")

    def test_plan_extracts_json_from_wrapped_text(self) -> None:
        llm = FakeLLM(
            output=(
                "Here is my routing decision:\n"
                '{"action":"recipe","reason":"food request",'
                '"params":{"ingredients":"eggs"},"clarifying_question":""}'
                "\nThanks"
            )
        )
        planner = Planner(llm=llm)

        plan = planner.plan(history=[], user_message="What can I cook with eggs?")

        self.assertEqual(plan.action, Action.RECIPE)
        self.assertEqual(plan.params, {"ingredients": "eggs"})
        self.assertIsNone(plan.clarifying_question)

    def test_plan_invalid_json_falls_back_to_clarify(self) -> None:
        llm = FakeLLM(output="not json")
        planner = Planner(llm=llm)

        plan = planner.plan(history=[], user_message="Do something")

        self.assertEqual(plan.action, Action.CLARIFY)
        self.assertEqual(plan.reason, "Planner output was not valid JSON")
        self.assertIn("joke or a recipe", plan.clarifying_question or "")

    def test_plan_invalid_action_and_params_are_sanitized(self) -> None:
        llm = FakeLLM(
            output=(
                '{"action":"unknown","reason":"",'
                '"params":[1,2,3],"clarifying_question":"   "}'
            )
        )
        planner = Planner(llm=llm)

        plan = planner.plan(history=[], user_message="Anything")

        self.assertEqual(plan.action, Action.CLARIFY)
        self.assertEqual(plan.reason, "No reason provided")
        self.assertEqual(plan.params, {})
        self.assertIsNone(plan.clarifying_question)


if __name__ == "__main__":
    unittest.main()
