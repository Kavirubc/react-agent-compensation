"""Strategy for deep recursive search through nested structures.

When heuristics fail on flat structures, this strategy searches
recursively through nested dicts and lists to find ID fields.
"""

from __future__ import annotations

from typing import Any

from react_agent_compensation.core.extraction.base import ExtractionStrategy, ToolLike


class RecursiveSearchStrategy(ExtractionStrategy):
    """Deep search through nested structures for ID fields.

    When heuristics fail on flat structures, this strategy searches
    recursively through nested dicts and lists to find ID fields.

    Example:
        result = {
            "data": {
                "booking": {
                    "id": "12345",
                    "details": {...}
                }
            }
        }
        # Returns: {"id": "12345"}
    """

    DEFAULT_ID_FIELDS: list[str] = [
        "id",
        "booking_id",
        "resource_id",
        "transaction_id",
        "order_id",
        "reservation_id",
    ]

    def __init__(
        self,
        max_depth: int = 5,
        id_fields: list[str] | None = None,
    ):
        """Initialize with search parameters.

        Args:
            max_depth: Maximum nesting depth to search (prevents infinite loops)
            id_fields: Custom list of ID field names to search for
        """
        self.max_depth = max_depth
        self.id_fields = id_fields or self.DEFAULT_ID_FIELDS

    def extract(
        self,
        result: Any,
        original_params: dict[str, Any],
        compensation_tool: ToolLike | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Extract parameters by recursively searching for ID fields.

        Args:
            result: The result from the original tool call
            original_params: The original parameters passed to the tool
            compensation_tool: The compensation tool (unused)
            tool_name: Name of the original tool (unused)

        Returns:
            Dict with first found ID field, or None if not found
        """
        if not isinstance(result, (dict, list)):
            return None

        found = self._search(result, depth=0, visited=set())
        return found if found else None

    def _search(
        self,
        data: Any,
        depth: int,
        visited: set[int],
    ) -> dict[str, Any] | None:
        """Recursively search for ID fields.

        Args:
            data: Current data to search
            depth: Current depth level
            visited: Set of object IDs already visited (cycle detection)

        Returns:
            Dict with found ID field, or None
        """
        if depth > self.max_depth:
            return None

        # Prevent infinite recursion on circular references
        data_id = id(data)
        if data_id in visited:
            return None
        visited.add(data_id)

        if isinstance(data, dict):
            # First check this level for ID fields
            for id_field in self.id_fields:
                if id_field in data and data[id_field]:
                    # Verify it looks like an ID (not empty, not a nested structure)
                    value = data[id_field]
                    if isinstance(value, (str, int)) and value:
                        return {id_field: value}

            # Then search nested structures
            for value in data.values():
                if isinstance(value, (dict, list)):
                    found = self._search(value, depth + 1, visited)
                    if found:
                        return found

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    found = self._search(item, depth + 1, visited)
                    if found:
                        return found

        return None
