"""Strategy for declarative schema-based extraction.

Second priority - allows developers to declare field mappings
without writing code.
"""

from __future__ import annotations

from typing import Any

from react_agent_compensation.core.extraction.base import (
    CompensationSchema,
    ExtractionStrategy,
    ToolLike,
)


class SchemaExtractionStrategy(ExtractionStrategy):
    """Use CompensationSchema for declarative extraction.

    Second priority - allows developers to declare field mappings
    without writing code.

    Example:
        schemas = {
            "book_flight": CompensationSchema(
                param_mapping={"booking_id": "result.id"},
                static_params={"reason": "Auto rollback"}
            )
        }
        strategy = SchemaExtractionStrategy(schemas)

        # Later, during compensation:
        params = strategy.extract(
            result={"id": "ABC123"},
            original_params={},
            tool_name="book_flight",
        )
        # Returns: {"booking_id": "ABC123", "reason": "Auto rollback"}
    """

    def __init__(self, schemas: dict[str, CompensationSchema] | None = None):
        """Initialize with schema mappings.

        Args:
            schemas: Dict mapping tool name -> CompensationSchema
        """
        self.schemas = schemas or {}

    def extract(
        self,
        result: Any,
        original_params: dict[str, Any],
        compensation_tool: ToolLike | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Extract parameters using declarative schema.

        Args:
            result: The result from the original tool call
            original_params: The original parameters passed to the tool
            compensation_tool: The compensation tool (unused)
            tool_name: Name of the original tool (for lookup)

        Returns:
            Extracted parameters dict, or None if no schema exists
        """
        if tool_name and tool_name in self.schemas:
            try:
                return self.schemas[tool_name].extract(result, original_params)
            except ValueError:
                # Schema extraction failed, defer to next strategy
                return None
        return None

    def add_schema(self, tool_name: str, schema: CompensationSchema) -> None:
        """Add or replace a schema for a tool.

        Args:
            tool_name: Name of the tool
            schema: The CompensationSchema instance
        """
        self.schemas[tool_name] = schema

    def remove_schema(self, tool_name: str) -> bool:
        """Remove a schema for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            True if schema was removed, False if it didn't exist
        """
        if tool_name in self.schemas:
            del self.schemas[tool_name]
            return True
        return False
