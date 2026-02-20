from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from agentic_chatbot.mcp import HttpMCPClient


@dataclass
class FakeTransport:
    response: dict[str, Any]
    last_url: str | None = None
    last_payload: dict[str, Any] | None = None
    last_timeout: float | None = None

    def post_json(self, url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        self.last_url = url
        self.last_payload = payload
        self.last_timeout = timeout
        return self.response


class MCPClientTests(unittest.TestCase):
    def test_call_tool_posts_expected_payload_and_endpoint(self) -> None:
        transport = FakeTransport(response={"result": " tool output "})
        client = HttpMCPClient(transport=transport, timeout_seconds=3.5)

        result = client.call_tool(
            server="https://mcp.example.com/",
            tool_name="search_docs",
            arguments={"q": "planner"},
        )

        self.assertEqual(result, "tool output")
        self.assertEqual(transport.last_url, "https://mcp.example.com/tools/call")
        self.assertEqual(
            transport.last_payload,
            {"tool_name": "search_docs", "arguments": {"q": "planner"}},
        )
        self.assertEqual(transport.last_timeout, 3.5)

    def test_call_tool_uses_content_blocks_when_result_is_missing(self) -> None:
        transport = FakeTransport(response={"content": [{"text": "first"}, {"text": "second"}]})
        client = HttpMCPClient(transport=transport)

        result = client.call_tool(server="https://mcp.example.com", tool_name="x", arguments={})

        self.assertEqual(result, "first\nsecond")

    def test_call_tool_raises_on_error_payload(self) -> None:
        transport = FakeTransport(response={"error": "tool failed"})
        client = HttpMCPClient(transport=transport)

        with self.assertRaises(ValueError):
            client.call_tool(server="https://mcp.example.com", tool_name="x", arguments={})


if __name__ == "__main__":
    unittest.main()
