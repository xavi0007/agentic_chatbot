from __future__ import annotations

from .factory import build_default_agent
from .schemas import ChatMessage, Role
from .mcp import MCPClient
import asyncio


async def main():
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
        history.append(ChatMessage(
            role=Role.ASSISTANT, content=response.content))

    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
