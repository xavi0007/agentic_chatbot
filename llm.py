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
        payload = [{"role": msg.role.value, "content": msg.content} for msg in messages]
        response = self.sdk_client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI response returned empty content")
        return content
