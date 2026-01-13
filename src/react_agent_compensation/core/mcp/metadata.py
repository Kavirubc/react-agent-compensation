"""MCP Tool Metadata model for annotation-driven compensation.

Provides structured representation of MCP tool annotations and helper
functions for determining compensation behavior based on metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCPToolMetadata:
    """Structured representation of MCP tool annotations.

    Captures all compensation-relevant annotations from MCP tool schemas:
    - action_type: create, read, update, delete
    - compensation_pair: linked compensator tool name
    - is_reversible: whether updates return previous state
    - is_destructive: dangerous operation flag
    - requires_confirmation: needs user approval flag
    - category: tool category for grouping
    """

    name: str
    action_type: str | None = None
    compensation_pair: str | None = None
    is_reversible: bool = False
    is_destructive: bool = False
    requires_confirmation: bool = False
    category: str | None = None
    raw_annotations: dict[str, Any] = field(default_factory=dict)


def parse_tool_metadata(schema: dict[str, Any]) -> MCPToolMetadata | None:
    """Extract all annotations from MCP tool schema.

    Searches for annotations in:
    1. inputSchema (x-* fields)
    2. annotations dict
    3. top-level schema fields

    Args:
        schema: MCP tool schema dict with name and inputSchema

    Returns:
        MCPToolMetadata if name found, None otherwise
    """
    tool_name = schema.get("name")
    if not tool_name:
        return None

    # collect all annotations from various locations
    annotations: dict[str, Any] = {}

    # check inputSchema for x-* fields
    input_schema = schema.get("inputSchema", {})
    if isinstance(input_schema, dict):
        for key, value in input_schema.items():
            if key.startswith("x-"):
                annotations[key] = value

    # check annotations dict (higher priority)
    schema_annotations = schema.get("annotations", {})
    if isinstance(schema_annotations, dict):
        for key, value in schema_annotations.items():
            # normalize keys to x-* format
            normalized_key = key if key.startswith("x-") else f"x-{key}"
            annotations[normalized_key] = value

    # check top-level for any x-* fields
    for key, value in schema.items():
        if key.startswith("x-") and key not in annotations:
            annotations[key] = value

    # extract specific fields with type coercion
    return MCPToolMetadata(
        name=tool_name,
        action_type=_get_string(annotations, "x-action-type"),
        compensation_pair=_get_string(annotations, "x-compensation-pair"),
        is_reversible=_get_bool(annotations, "x-reversible"),
        is_destructive=_get_bool(annotations, "x-destructive"),
        requires_confirmation=_get_bool(annotations, "x-requires-confirmation"),
        category=_get_string(annotations, "x-category"),
        raw_annotations=annotations,
    )


def _get_string(annotations: dict[str, Any], key: str) -> str | None:
    """Get string value from annotations."""
    value = annotations.get(key)
    if value is not None and isinstance(value, str):
        return value
    return None


def _get_bool(annotations: dict[str, Any], key: str) -> bool:
    """Get boolean value from annotations."""
    value = annotations.get(key)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def should_track_action(metadata: MCPToolMetadata) -> bool:
    """Determine if action should be tracked for compensation.

    Read operations don't need tracking since they have no side effects.

    Args:
        metadata: Tool metadata

    Returns:
        False for read operations, True otherwise
    """
    if metadata.action_type == "read":
        return False
    return True


def get_compensator(metadata: MCPToolMetadata) -> str | None:
    """Get compensator tool name from metadata.

    For reversible updates, the compensator is the tool itself
    (self-compensating via previous_* fields).

    Args:
        metadata: Tool metadata

    Returns:
        Compensator tool name or None if not compensatable
    """
    # reversible updates are self-compensating
    if metadata.is_reversible and metadata.action_type == "update":
        return metadata.name

    return metadata.compensation_pair


def is_compensatable(metadata: MCPToolMetadata) -> bool:
    """Check if tool has compensation capability.

    Args:
        metadata: Tool metadata

    Returns:
        True if tool can be compensated
    """
    return get_compensator(metadata) is not None
