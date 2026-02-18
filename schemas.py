from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class Action(str, Enum):
    CLARIFY = "clarify"
    JOKE = "joke"
    RECIPE = "recipe"


@dataclass(frozen=True)
class Plan:
    action: Action
    reason: str
    params: dict[str, Any] = field(default_factory=dict)
    clarifying_question: str | None = None


@dataclass(frozen=True)
class AgentResponse:
    content: str
    action: Action
