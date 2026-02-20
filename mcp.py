from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class MCPClient(Protocol):
    def call_tool(self, *, server: str, tool_name: str, arguments: dict[str, Any]) -> str:
        ...

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
