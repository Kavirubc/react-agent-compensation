"""MCP (Model Context Protocol) integration for compensation discovery.

This module provides functions to parse MCP tool schemas and discover
compensation pairs automatically from x-compensation-pair extension fields.

Components:
- MCPCompensationClient: High-level client with auto compensation discovery
- CompensatedMCPTool: Tool wrapper with compensation tracking
- parse_mcp_schema(): Extract compensation pair from single schema
- discover_compensation_pairs(): Scan multiple tools for pairs
- register_from_mcp(): Register discovered pairs with RecoveryManager
- validate_mcp_schema(): Validate schema for compensation readiness

Example:
    from react_agent_compensation.core.mcp import MCPCompensationClient

    client = MCPCompensationClient({
        "family": {"url": "http://localhost:8000/sse", "transport": "sse"}
    })
    await client.connect()
    tools = await client.get_tools()  # Wrapped with compensation tracking
"""

from react_agent_compensation.core.mcp.client import (
    MCPCompensationClient,
    MCPToolExecutor,
)
from react_agent_compensation.core.mcp.parser import (
    discover_compensation_pairs,
    parse_mcp_schema,
    register_from_mcp,
    validate_mcp_schema,
)
from react_agent_compensation.core.mcp.tools import (
    CompensatedMCPTool,
    MCPToolError,
    wrap_mcp_tools,
)

__all__ = [
    # Client
    "MCPCompensationClient",
    "MCPToolExecutor",
    # Tools
    "CompensatedMCPTool",
    "MCPToolError",
    "wrap_mcp_tools",
    # Parser functions
    "parse_mcp_schema",
    "discover_compensation_pairs",
    "register_from_mcp",
    "validate_mcp_schema",
]
