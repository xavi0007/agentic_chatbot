from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


class MCPClient(Protocol):
    def call_tool(self, *, server: str, tool_name: str, arguments: dict[str, Any]) -> str:
        ... 


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

    def get_prompt(
        self,
        *,
        server: str,
        prompt_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        ...


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
    prompt_connectors: dict[str, MCPPromptConnector] = field(default_factory=dict)

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
