"""Strategy for heuristic-based ID field extraction.

This is the default strategy for most tools - looks for common
field names that typically contain identifiers needed for
compensation.
"""

from __future__ import annotations

from typing import Any

from react_agent_compensation.core.extraction.base import ExtractionStrategy, ToolLike


class HeuristicExtractionStrategy(ExtractionStrategy):
    """Extract common ID fields using heuristics.

    This is the default strategy for most tools - looks for common
    field names that typically contain identifiers needed for
    compensation.

    Priority of fields checked (configurable):
    1. id
    2. booking_id
    3. resource_id
    4. transaction_id
    5. order_id
    6. reservation_id
    7. confirmation_id
    8. reference_id
    9. request_id

    Example:
        result = {"id": "12345", "status": "confirmed"}
        # Returns: {"id": "12345"}

        result = "ABC123"  # String result
        # Returns: {"id": "ABC123"}
    """

    # Default common ID field names in order of priority
    DEFAULT_ID_FIELDS: list[str] = [
        "id",
        "booking_id",
        "resource_id",
        "transaction_id",
        "order_id",
        "reservation_id",
        "confirmation_id",
        "reference_id",
        "request_id",
        "record_id",
        "item_id",
        "entity_id",
    ]

    def __init__(self, id_fields: list[str] | None = None):
        """Initialize with configurable ID field names.

        Args:
            id_fields: Custom list of ID field names to check, in priority order.
                If None, uses DEFAULT_ID_FIELDS.
        """
        self.id_fields = id_fields or self.DEFAULT_ID_FIELDS

    def extract(
        self,
        result: Any,
        original_params: dict[str, Any],
        compensation_tool: ToolLike | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Extract parameters using heuristic field matching.

        Args:
            result: The result from the original tool call
            original_params: The original parameters passed to the tool
            compensation_tool: The compensation tool (unused)
            tool_name: Name of the original tool (unused)

        Returns:
            Dict with first matching ID field, or None if no match
        """
        if not isinstance(result, dict):
            # For string results, assume it's the ID itself
            if isinstance(result, str) and result:
                return {"id": result}
            # For integer results, could be an ID
            if isinstance(result, int) and result > 0:
                return {"id": result}
            return None

        # Look for common ID fields in priority order
        for id_field in self.id_fields:
            if id_field in result and result[id_field]:
                return {id_field: result[id_field]}

        return None

    def add_id_field(self, field_name: str, priority: int | None = None) -> None:
        """Add an ID field to check.

        Args:
            field_name: Name of the field to add
            priority: Index to insert at (0 = highest priority).
                If None, appends to end.
        """
        if field_name not in self.id_fields:
            if priority is not None:
                self.id_fields.insert(priority, field_name)
            else:
                self.id_fields.append(field_name)
