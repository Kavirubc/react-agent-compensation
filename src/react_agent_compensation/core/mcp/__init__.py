"""MCP (Model Context Protocol) integration for compensation discovery.

This module provides functions to parse MCP tool schemas and discover
compensation pairs automatically from x-compensation-pair extension fields.

Functions:
- parse_mcp_schema() - Extract compensation pair from single schema
- discover_compensation_pairs() - Scan multiple tools for pairs
- register_from_mcp() - Register discovered pairs with RecoveryManager
- validate_mcp_schema() - Validate schema for compensation readiness
"""

from react_agent_compensation.core.mcp.parser import (
    discover_compensation_pairs,
    parse_mcp_schema,
    register_from_mcp,
    validate_mcp_schema,
)

__all__ = [
    "parse_mcp_schema",
    "discover_compensation_pairs",
    "register_from_mcp",
    "validate_mcp_schema",
]
