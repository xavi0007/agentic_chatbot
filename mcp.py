from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env


# Why use MCP
# Don't bloat the agent with tool-specific logic and intermediary steps- keep it focused on reasoning and decision-making
# Decouple tool execution from agent reasoning - allows for more flexible and powerful tool interactions
# Support more complex tool interactions, like multi-step calls, conditional logic based on tool results, and dynamic tool discovery

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
    # methods will go here

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        final_text = []

        assistant_message_content = []
        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(
                    f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }
                    ]
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:",
              [tool.name for tool in tools])

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


class HTTPTransport(Protocol):
    def post_json(self, url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class UrllibHTTPTransport:
    def post_json(self, url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"MCP HTTP {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            raise ValueError(f"MCP network error: {exc.reason}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("MCP response was not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("MCP response must be a JSON object")
        return parsed


@dataclass
class HttpMCPClient:
    transport: HTTPTransport = field(default_factory=UrllibHTTPTransport)
    timeout_seconds: float = 15.0

    def call_tool(self, *, server: str, tool_name: str, arguments: dict[str, Any]) -> str:
        payload = {
            "tool_name": tool_name,
            "arguments": arguments,
        }
        response = self.transport.post_json(
            _join_endpoint(server, "/tools/call"),
            payload,
            timeout=self.timeout_seconds,
        )
        return _extract_text(response, "result")

    def get_prompt(
        self,
        *,
        server: str,
        prompt_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "prompt_name": prompt_name,
            "arguments": arguments or {},
        }
        response = self.transport.post_json(
            _join_endpoint(server, "/prompts/get"),
            payload,
            timeout=self.timeout_seconds,
        )
        return _extract_text(response, "prompt")


@dataclass(frozen=True)
class MCPToolConnector:
    client: MCPClient
    server: str
    tool_name: str
    default_arguments: dict[str, Any] = field(default_factory=dict)

    def run(self, arguments: dict[str, Any] | None = None) -> str:
        merged_args = {**self.default_arguments, **(arguments or {})}
        return self.client.call_tool(
            server=self.server,
            tool_name=self.tool_name,
            arguments=merged_args,
        )


@dataclass(frozen=True)
class MCPPromptConnector:
    client: MCPClient
    server: str
    prompt_name: str
    default_arguments: dict[str, Any] = field(default_factory=dict)

    def resolve(self, arguments: dict[str, Any] | None = None) -> str:
        merged_args = {**self.default_arguments, **(arguments or {})}
        return self.client.get_prompt(
            server=self.server,
            prompt_name=self.prompt_name,
            arguments=merged_args,
        )


@dataclass
class MCPConnectorRegistry:
    tool_connectors: dict[str, MCPToolConnector] = field(default_factory=dict)
    prompt_connectors: dict[str, MCPPromptConnector] = field(
        default_factory=dict)

    def register_tool(self, alias: str, connector: MCPToolConnector) -> None:
        self.tool_connectors[alias] = connector

    def register_prompt(self, alias: str, connector: MCPPromptConnector) -> None:
        self.prompt_connectors[alias] = connector

    def call_tool(self, alias: str, arguments: dict[str, Any] | None = None) -> str:
        connector = self.tool_connectors.get(alias)
        if connector is None:
            raise KeyError(f"MCP tool connector not found: {alias}")
        return connector.run(arguments)

    def get_prompt(self, alias: str, arguments: dict[str, Any] | None = None) -> str:
        connector = self.prompt_connectors.get(alias)
        if connector is None:
            raise KeyError(f"MCP prompt connector not found: {alias}")
        return connector.resolve(arguments)


def _join_endpoint(server: str, endpoint: str) -> str:
    base = server.strip().rstrip("/")
    if not base:
        raise ValueError("MCP server URL cannot be empty")
    return f"{base}{endpoint}"


def _extract_text(payload: dict[str, Any], primary_key: str) -> str:
    if "error" in payload and payload["error"]:
        raise ValueError(f"MCP error: {payload['error']}")

    value = payload.get(primary_key)
    if isinstance(value, str) and value.strip():
        return value.strip()

    # Some MCP servers return structured blocks with text in `content`.
    content_blocks = payload.get("content")
    if isinstance(content_blocks, list):
        texts: list[str] = []
        for item in content_blocks:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        if texts:
            return "\n".join(texts)

    raise ValueError("MCP response missing textual content")
