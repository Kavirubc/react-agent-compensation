"""MCP (Model Context Protocol) integration for compensation discovery.

This module provides functions to parse MCP tool schemas and discover
compensation pairs from custom extension fields.

MCP tools can declare compensation relationships using:
- x-compensation-pair in inputSchema
- x-compensation in annotations

Example MCP tool schema with compensation:
    {
        "name": "book_flight",
        "description": "Book a flight",
        "inputSchema": {
            "type": "object",
            "x-compensation-pair": "cancel_flight",
            "properties": {...}
        }
    }
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from react_agent_compensation.core.protocols import ToolSchemaProvider
    from react_agent_compensation.core.recovery_manager import RecoveryManager


logger = logging.getLogger(__name__)


def parse_mcp_schema(schema: dict[str, Any]) -> tuple[str, str] | None:
    """Extract compensation pair from MCP tool schema.

    Looks for x-compensation-pair in:
    1. Top-level schema
    2. inputSchema
    3. annotations

    Args:
        schema: MCP tool schema dict

    Returns:
        Tuple of (tool_name, compensation_tool) if found, None otherwise

    Example:
        schema = {
            "name": "book_flight",
            "inputSchema": {
                "x-compensation-pair": "cancel_flight"
            }
        }
        result = parse_mcp_schema(schema)
        # Returns: ("book_flight", "cancel_flight")
    """
    tool_name = schema.get("name")
    if not tool_name:
        return None

    # Check inputSchema for x-compensation-pair
    input_schema = schema.get("inputSchema", {})
    if isinstance(input_schema, dict):
        comp_pair = input_schema.get("x-compensation-pair")
        if comp_pair and isinstance(comp_pair, str):
            return (tool_name, comp_pair)

    # Check top-level x-compensation-pair
    comp_pair = schema.get("x-compensation-pair")
    if comp_pair and isinstance(comp_pair, str):
        return (tool_name, comp_pair)

    # Check annotations
    annotations = schema.get("annotations", {})
    if isinstance(annotations, dict):
        comp_pair = annotations.get("x-compensation-pair")
        if comp_pair and isinstance(comp_pair, str):
            return (tool_name, comp_pair)

        # Also check for compensation in different format
        comp_pair = annotations.get("compensation")
        if comp_pair and isinstance(comp_pair, str):
            return (tool_name, comp_pair)

    return None


def discover_compensation_pairs(
    tools: list["ToolSchemaProvider"] | list[dict[str, Any]],
) -> dict[str, str]:
    """Auto-discover compensation pairs from tool schemas.

    Scans a list of tools for x-compensation-pair declarations
    and returns a mapping of tool names to their compensators.

    Args:
        tools: List of tools (either ToolSchemaProvider objects or dicts)

    Returns:
        Dict mapping tool names to compensation tool names

    Example:
        tools = [
            {"name": "book_flight", "inputSchema": {"x-compensation-pair": "cancel_flight"}},
            {"name": "book_hotel", "inputSchema": {"x-compensation-pair": "cancel_hotel"}},
        ]
        pairs = discover_compensation_pairs(tools)
        # Returns: {"book_flight": "cancel_flight", "book_hotel": "cancel_hotel"}
    """
    pairs: dict[str, str] = {}

    for tool in tools:
        schema = _get_tool_schema(tool)
        if schema:
            result = parse_mcp_schema(schema)
            if result:
                tool_name, comp_tool = result
                pairs[tool_name] = comp_tool
                logger.debug(f"Discovered compensation pair: {tool_name} -> {comp_tool}")

    return pairs


def _get_tool_schema(tool: Any) -> dict[str, Any] | None:
    """Extract schema dict from a tool object.

    Args:
        tool: Tool object (dict or ToolSchemaProvider)

    Returns:
        Schema dict or None
    """
    if isinstance(tool, dict):
        return tool

    # Try ToolSchemaProvider protocol
    if hasattr(tool, "name") and hasattr(tool, "get_input_schema"):
        try:
            return {
                "name": tool.name,
                "description": getattr(tool, "description", ""),
                "inputSchema": tool.get_input_schema(),
            }
        except Exception:
            pass

    # Try LangChain-style tool
    if hasattr(tool, "name") and hasattr(tool, "args_schema"):
        try:
            args_schema = tool.args_schema
            if args_schema and hasattr(args_schema, "schema"):
                return {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "inputSchema": args_schema.schema(),
                }
        except Exception:
            pass

    return None


def register_from_mcp(
    manager: "RecoveryManager",
    tools: list["ToolSchemaProvider"] | list[dict[str, Any]],
) -> int:
    """Register discovered compensation pairs with a RecoveryManager.

    Scans tools for compensation declarations and adds them to the
    manager's compensation pairs.

    Args:
        manager: RecoveryManager to register pairs with
        tools: List of tools to scan

    Returns:
        Number of pairs registered

    Example:
        manager = RecoveryManager(compensation_pairs={})
        count = register_from_mcp(manager, tools)
        print(f"Registered {count} compensation pairs from MCP schemas")
    """
    pairs = discover_compensation_pairs(tools)

    for tool_name, comp_tool in pairs.items():
        manager.add_compensation_pair(tool_name, comp_tool)
        logger.info(f"Registered MCP compensation pair: {tool_name} -> {comp_tool}")

    return len(pairs)


def validate_mcp_schema(schema: dict[str, Any]) -> list[str]:
    """Validate an MCP tool schema for compensation readiness.

    Checks that:
    1. Schema has required fields (name)
    2. If compensation declared, the compensator tool name is valid
    3. inputSchema is well-formed

    Args:
        schema: MCP tool schema to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors: list[str] = []

    # Check required name
    if "name" not in schema:
        errors.append("Missing required field: name")
        return errors

    tool_name = schema["name"]
    if not isinstance(tool_name, str) or not tool_name:
        errors.append("Field 'name' must be a non-empty string")

    # Check inputSchema
    input_schema = schema.get("inputSchema")
    if input_schema is not None:
        if not isinstance(input_schema, dict):
            errors.append("Field 'inputSchema' must be an object")
        else:
            # Check x-compensation-pair if present
            comp_pair = input_schema.get("x-compensation-pair")
            if comp_pair is not None:
                if not isinstance(comp_pair, str) or not comp_pair:
                    errors.append("Field 'x-compensation-pair' must be a non-empty string")

    return errors
