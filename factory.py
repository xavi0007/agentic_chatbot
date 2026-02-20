from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from .agent import AgenticChatbot
from .llm import AnthropicChatClient, GoogleChatClient, OpenAIChatClient
from .mcp import MCPConnectorRegistry
from .planner import Planner
from .skills import ClarifySkill, JokeSkill, RecipeSkill

Provider = Literal["openai", "anthropic", "google"]


@dataclass
class ChatbotFactory:
    provider: Provider = "openai"
    model: str | None = None
    api_key: str | None = None
    sdk_client: Any | None = None
    mcp_registry: MCPConnectorRegistry | None = None

    @classmethod
    def from_env(cls) -> "ChatbotFactory":
        provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        if provider not in {"openai", "anthropic", "google"}:
            provider = "openai"

        env_model_key = {
            "openai": "OPENAI_MODEL",
            "anthropic": "ANTHROPIC_MODEL",
            "google": "GOOGLE_MODEL",
        }[provider]
        env_key_key = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }[provider]

        return cls(
            provider=provider,  # type: ignore[arg-type]
            model=os.getenv(env_model_key),
            api_key=os.getenv(env_key_key),
        )

    def build_llm(self):
        provider = self.provider.strip().lower()
        if provider == "openai":
            return OpenAIChatClient(
                model=self.model or "gpt-5-mini",
                api_key=self.api_key,
                sdk_client=self.sdk_client,
            )
        if provider == "anthropic":
            return AnthropicChatClient(
                model=self.model or "claude-3-5-sonnet-latest",
                api_key=self.api_key,
                sdk_client=self.sdk_client,
            )
        if provider == "google":
            return GoogleChatClient(
                model=self.model or "gemini-2.5-flash",
                api_key=self.api_key,
                sdk_client=self.sdk_client,
            )
        raise ValueError(f"Unsupported provider: {self.provider}")

    def build_agent(self) -> AgenticChatbot:
        llm = self.build_llm()
        planner = Planner(llm=llm)

        return AgenticChatbot(
            planner=planner,
            clarify_skill=ClarifySkill(),
            joke_skill=JokeSkill(llm=llm, mcp_registry=self.mcp_registry),
            recipe_skill=RecipeSkill(llm=llm, mcp_registry=self.mcp_registry),
        )


def build_default_agent() -> AgenticChatbot:
    return ChatbotFactory.from_env().build_agent()
