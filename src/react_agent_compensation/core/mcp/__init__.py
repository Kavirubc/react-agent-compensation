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
from react_agent_compensation.core.mcp.metadata import (
    MCPToolMetadata,
    get_compensator,
    is_compensatable,
    parse_tool_metadata,
    should_track_action,
)
from react_agent_compensation.core.mcp.parser import (
    build_compensation_pairs_from_metadata,
    discover_compensation_pairs,
    discover_tool_metadata,
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
    # Metadata
    "MCPToolMetadata",
    "parse_tool_metadata",
    "should_track_action",
    "get_compensator",
    "is_compensatable",
    # Tools
    "CompensatedMCPTool",
    "MCPToolError",
    "wrap_mcp_tools",
    # Parser functions
    "parse_mcp_schema",
    "discover_compensation_pairs",
    "discover_tool_metadata",
    "build_compensation_pairs_from_metadata",
    "register_from_mcp",
    "validate_mcp_schema",
]
