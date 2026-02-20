from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentic_chatbot.schemas import Action, ChatMessage, Plan, Role
from agentic_chatbot.skills import ClarifySkill, JokeSkill, RecipeSkill


@dataclass
class RecordingLLM:
    response_text: str = "ok"
    last_messages: list[ChatMessage] = field(default_factory=list)
    last_temperature: float | None = None

    def complete(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> str:
        self.last_messages = messages
        self.last_temperature = temperature
        return self.response_text


@dataclass
class FakeRegistry:
    prompt_text: str = ""
    tool_text: str = ""
    raise_prompt: bool = False
    raise_tool: bool = False
    last_prompt_alias: str | None = None
    last_prompt_args: dict[str, Any] | None = None
    last_tool_alias: str | None = None
    last_tool_args: dict[str, Any] | None = None

    def get_prompt(self, alias: str, arguments: dict[str, Any] | None = None) -> str:
        if self.raise_prompt:
            raise KeyError("missing prompt")
        self.last_prompt_alias = alias
        self.last_prompt_args = arguments or {}
        return self.prompt_text

    def call_tool(self, alias: str, arguments: dict[str, Any] | None = None) -> str:
        if self.raise_tool:
            raise ValueError("tool failed")
        self.last_tool_alias = alias
        self.last_tool_args = arguments or {}
        return self.tool_text


class SkillTests(unittest.TestCase):
    def test_clarify_skill_prefers_plan_question(self) -> None:
        skill = ClarifySkill()
        plan = Plan(action=Action.CLARIFY, reason="ambiguous", clarifying_question="Can you clarify?")

        result = skill.run(plan, history=[], user_message="help")

        self.assertEqual(result.action, Action.CLARIFY)
        self.assertEqual(result.content, "Can you clarify?")

    def test_joke_skill_builds_prompt_and_uses_temperature(self) -> None:
        llm = RecordingLLM(response_text="  A clean cat joke.  ")
        skill = JokeSkill(llm=llm)
        history = [ChatMessage(role=Role.USER, content="Previous context")]
        plan = Plan(action=Action.JOKE, reason="joke", params={"topic": "cats", "style": "dry"})

        result = skill.run(plan, history=history, user_message="Tell me a joke")

        self.assertEqual(result.action, Action.JOKE)
        self.assertEqual(result.content, "A clean cat joke.")
        self.assertEqual(llm.last_temperature, 0.8)
        self.assertEqual(llm.last_messages[0].content, "You are a concise comedian.")
        self.assertTrue(llm.last_messages[-1].content.startswith("Write one short joke."))
        self.assertIn("Topic: cats. Style: dry.", llm.last_messages[-1].content)

    def test_joke_skill_uses_mcp_prompt_and_tool_context(self) -> None:
        llm = RecordingLLM(response_text="Joke")
        registry = FakeRegistry(prompt_text="Custom comedian prompt", tool_text="Facts from tool")
        skill = JokeSkill(llm=llm, mcp_registry=registry)
        plan = Plan(
            action=Action.JOKE,
            reason="joke",
            params={
                "mcp_prompt": "comedy_prompt",
                "prompt_args": {"tone": "playful"},
                "mcp_tool": "fact_tool",
                "tool_args": {"region": "us"},
            },
        )

        skill.run(plan, history=[], user_message="Tell me a joke about weather")

        self.assertEqual(llm.last_messages[0].content, "Custom comedian prompt")
        self.assertIn("External tool context:\nFacts from tool", llm.last_messages[-1].content)
        self.assertEqual(registry.last_prompt_alias, "comedy_prompt")
        self.assertEqual(registry.last_prompt_args, {"tone": "playful", "user_message": "Tell me a joke about weather"})
        self.assertEqual(registry.last_tool_alias, "fact_tool")
        self.assertEqual(registry.last_tool_args, {"region": "us", "user_message": "Tell me a joke about weather"})

    def test_recipe_skill_falls_back_when_mcp_fails(self) -> None:
        llm = RecordingLLM(response_text="Recipe")
        registry = FakeRegistry(raise_prompt=True, raise_tool=True)
        skill = RecipeSkill(llm=llm, mcp_registry=registry)
        plan = Plan(
            action=Action.RECIPE,
            reason="food",
            params={"mcp_prompt": "broken", "mcp_tool": "broken_tool", "ingredients": "rice"},
        )

        result = skill.run(plan, history=[], user_message="Make rice")

        self.assertEqual(result.action, Action.RECIPE)
        self.assertEqual(llm.last_messages[0].content, "You are a precise cooking assistant.")
        self.assertNotIn("External tool context", llm.last_messages[-1].content)
        self.assertIn("Ingredients preference: rice.", llm.last_messages[-1].content)


if __name__ == "__main__":
    unittest.main()
