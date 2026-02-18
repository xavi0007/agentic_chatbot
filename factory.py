from __future__ import annotations

import os

from .agent import AgenticChatbot
from .llm import OpenAIChatClient
from .planner import Planner
from .skills import ClarifySkill, JokeSkill, RecipeSkill


def build_default_agent() -> AgenticChatbot:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    llm = OpenAIChatClient(model=model, api_key=api_key)
    planner = Planner(llm=llm)

    return AgenticChatbot(
        planner=planner,
        clarify_skill=ClarifySkill(),
        joke_skill=JokeSkill(llm=llm),
        recipe_skill=RecipeSkill(llm=llm),
    )
