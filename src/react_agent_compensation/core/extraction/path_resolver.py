"""Path resolution for dot notation access in compensation schemas.

This module provides functions to resolve dot-notation paths like
"result.nested.field" or "result.items[0].id" to actual values.
"""

from __future__ import annotations

import re
from typing import Any


def resolve_path(path: str, context: dict[str, Any]) -> Any:
    """Resolve a dot-notation path to a value.

    Supports:
        - Dot notation: "result.nested.field"
        - Array indices: "result.items[0]"
        - Mixed: "result.items[0].name"
        - Nested arrays: "result.data[0][1]"

    Args:
        path: Dot-notation path expression
        context: Dict containing "result" and "params" keys

    Returns:
        Resolved value

    Raises:
        KeyError: If path segment not found
        IndexError: If array index out of bounds
        TypeError: If trying to access non-subscriptable type

    Examples:
        >>> context = {"result": {"id": 123, "items": [{"name": "a"}]}}
        >>> resolve_path("result.id", context)
        123
        >>> resolve_path("result.items[0].name", context)
        'a'
    """
    # Split on dots, but preserve array indices
    # "result.items[0].name" -> ["result", "items[0]", "name"]
    parts = re.split(r"\.(?![^\[]*\])", path)
    current = context

    for part in parts:
        current = _resolve_part(part, current)

    return current


def _resolve_part(part: str, current: Any) -> Any:
    """Resolve a single path part which may contain array indices.

    Args:
        part: Path segment like "items" or "items[0]" or "items[0][1]"
        current: Current value to access

    Returns:
        Resolved value for this segment

    Raises:
        KeyError: If key not found
        IndexError: If array index out of bounds
        TypeError: If access not supported
    """
    # Check for array indices: "items[0]" or "items[0][1]"
    # Pattern matches: key followed by one or more [index] parts
    match = re.match(r"(\w+)((?:\[\d+\])+)?", part)

    if not match:
        raise KeyError(f"Invalid path segment: {part}")

    key, indices_str = match.groups()

    # First access the key
    if isinstance(current, dict):
        if key not in current:
            raise KeyError(f"Key '{key}' not found in dict")
        current = current[key]
    elif hasattr(current, key):
        current = getattr(current, key)
    else:
        raise KeyError(f"Cannot access '{key}' on {type(current).__name__}")

    # Then apply any array indices
    if indices_str:
        indices = re.findall(r"\[(\d+)\]", indices_str)
        for idx in indices:
            if not isinstance(current, (list, tuple)):
                raise TypeError(f"Cannot index {type(current).__name__} with [{idx}]")
            current = current[int(idx)]

    return current


def validate_path(path: str) -> bool:
    """Validate that a path expression is syntactically correct.

    Args:
        path: Path expression to validate

    Returns:
        True if valid, False otherwise

    Examples:
        >>> validate_path("result.id")
        True
        >>> validate_path("result.items[0].name")
        True
        >>> validate_path("result.field?")
        True
        >>> validate_path("invalid..path")
        False
    """
    pattern = r"^(\w+(\[\d+\])*)(\.(\w+(\[\d+\])*))*\??$"
    return bool(re.match(pattern, path))


def extract_all_values(data: Any, max_depth: int = 10) -> set[Any]:
    """Recursively extract all primitive values from a nested structure.

    Used for dependency inference - extracts all leaf values that could
    potentially be IDs or references.

    Args:
        data: The data structure to extract values from
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        Set of all primitive values found
    """
    values: set[Any] = set()

    def _extract(obj: Any, depth: int) -> None:
        if depth > max_depth:
            return

        if obj is None:
            return
        elif isinstance(obj, (str, int, float)):
            # Filter out noise values
            if isinstance(obj, bool):
                return  # Skip booleans
            if isinstance(obj, (int, float)) and abs(obj) < 10000:
                return  # Skip small numbers
            if isinstance(obj, str) and len(obj) < 5:
                return  # Skip short strings
            values.add(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _extract(v, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _extract(item, depth + 1)

    _extract(data, 0)
    return values
