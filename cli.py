from __future__ import annotations

from .factory import build_default_agent
from .schemas import ChatMessage, Role


def main() -> None:
    agent = build_default_agent()
    history: list[ChatMessage] = []

    print("Agentic chatbot ready. Type 'exit' to quit.")
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        response = agent.respond(history=history, user_message=user_input)
        print(f"assistant> {response.content}")

        history.append(ChatMessage(role=Role.USER, content=user_input))
        history.append(ChatMessage(role=Role.ASSISTANT, content=response.content))


if __name__ == "__main__":
    main()
