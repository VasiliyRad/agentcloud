"""
Adapts AgentCloud platform tools to AG2-compatible Python functions.
Platform tools are expected as { name, description, fn } dicts.
AG2 tools are plain callables registered via register_for_llm / register_for_execution.

AG2 also ships a native MCP client (from autogen.mcp import MCPClient, ag2[mcp] extra)
that connects to external MCP servers over SSE transport with no additional adapters.
This can be wired into AG2Builder in a follow-up once the platform surfaces MCP server
configuration in the app config UI.
"""
from __future__ import annotations

from typing import Callable


def build_ag2_tools(tool_configs: list[dict]) -> list[tuple[Callable, dict]]:
    """
    Convert platform tool configs to (callable, metadata) tuples for AG2 registration.
    Each entry in tool_configs must have keys: name, description, fn.
    """
    result = []
    for tc in tool_configs:
        fn = tc["fn"]
        meta = {"name": tc["name"], "description": tc["description"]}
        result.append((fn, meta))
    return result
