from __future__ import annotations

import os
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentic_chatbot.factory import ChatbotFactory
from agentic_chatbot.llm import AnthropicChatClient, GoogleChatClient, OpenAIChatClient


class FactoryTests(unittest.TestCase):
    def test_factory_from_env_reads_provider_specific_values(self) -> None:
        old = dict(os.environ)
        try:
            os.environ["LLM_PROVIDER"] = "anthropic"
            os.environ["ANTHROPIC_MODEL"] = "claude-custom"
            os.environ["ANTHROPIC_API_KEY"] = "anthropic-key"

            factory = ChatbotFactory.from_env()

            self.assertEqual(factory.provider, "anthropic")
            self.assertEqual(factory.model, "claude-custom")
            self.assertEqual(factory.api_key, "anthropic-key")
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_factory_from_env_invalid_provider_defaults_to_openai(self) -> None:
        old = dict(os.environ)
        try:
            os.environ["LLM_PROVIDER"] = "invalid"
            os.environ["OPENAI_MODEL"] = "gpt-custom"
            os.environ["OPENAI_API_KEY"] = "openai-key"

            factory = ChatbotFactory.from_env()

            self.assertEqual(factory.provider, "openai")
            self.assertEqual(factory.model, "gpt-custom")
            self.assertEqual(factory.api_key, "openai-key")
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_build_llm_uses_expected_default_models(self) -> None:
        sdk = object()

        openai_client = ChatbotFactory(provider="openai", sdk_client=sdk).build_llm()
        anthropic_client = ChatbotFactory(provider="anthropic", sdk_client=sdk).build_llm()
        google_client = ChatbotFactory(provider="google", sdk_client=sdk).build_llm()

        self.assertIsInstance(openai_client, OpenAIChatClient)
        self.assertIsInstance(anthropic_client, AnthropicChatClient)
        self.assertIsInstance(google_client, GoogleChatClient)

        self.assertEqual(openai_client.model, "gpt-5-mini")
        self.assertEqual(anthropic_client.model, "claude-3-5-sonnet-latest")
        self.assertEqual(google_client.model, "gemini-2.5-flash")


if __name__ == "__main__":
    unittest.main()
