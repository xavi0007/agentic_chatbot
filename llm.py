from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .schemas import ChatMessage


class LLMClient(Protocol):
    def complete(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> str:
        ...


@dataclass
class OpenAIChatClient:
    model: str
    api_key: str | None = None
    sdk_client: Any | None = None

    def __post_init__(self) -> None:
        if self.sdk_client is None:
            from openai import OpenAI

            self.sdk_client = OpenAI(api_key=self.api_key)

    def complete(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> str:
        payload = [{"role": msg.role.value, "content": msg.content}
                   for msg in messages]
        response = self.sdk_client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI response returned empty content")
        return content


@dataclass
class AnthropicChatClient:
    model: str
    api_key: str | None = None
    sdk_client: Any | None = None
    max_tokens: int = 512

    def __post_init__(self) -> None:
        if self.sdk_client is None:
            from anthropic import Anthropic

            self.sdk_client = Anthropic(api_key=self.api_key)

    def complete(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> str:
        system_parts = [msg.content for msg in messages if msg.role.value == "system"]
        convo_parts = [f"{msg.role.value}: {msg.content}" for msg in messages if msg.role.value != "system"]
        response = self.sdk_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system="\n".join(system_parts) if system_parts else None,
            messages=[{"role": "user", "content": "\n".join(convo_parts)}],
        )
        text_chunks: list[str] = []
        for chunk in response.content:
            text = getattr(chunk, "text", None)
            if text:
                text_chunks.append(text)
        content = "".join(text_chunks).strip()
        if not content:
            raise ValueError("Anthropic response returned empty content")
        return content


@dataclass
class GoogleChatClient:
    model: str
    api_key: str | None = None
    sdk_client: Any | None = None

    def __post_init__(self) -> None:
        if self.sdk_client is None:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.sdk_client = genai.GenerativeModel(self.model)

    def complete(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> str:
        prompt = "\n".join(f"{msg.role.value}: {msg.content}" for msg in messages)
        response = self.sdk_client.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
        content = getattr(response, "text", None)
        if content is None and getattr(response, "candidates", None):
            parts = response.candidates[0].content.parts
            content = "".join(getattr(part, "text", "") for part in parts)
        content = (content or "").strip()
        if not content:
            raise ValueError("Google response returned empty content")
        return content
