"""Strategy for developer-provided state mapper functions.

This is the highest priority strategy - if a developer provides
an explicit mapping function, use it.
"""

from __future__ import annotations

from typing import Any, Callable

from react_agent_compensation.core.extraction.base import ExtractionStrategy, ToolLike


# Type alias for state mapper functions
StateMapperFn = Callable[[Any, dict[str, Any]], dict[str, Any]]


class StateMappersStrategy(ExtractionStrategy):
    """Use developer-provided state_mapper functions.

    This is the highest priority strategy - if a developer provides
    an explicit mapping function, use it.

    Example:
        state_mappers = {
            "book_flight": lambda result, params: {
                "booking_id": result["booking"]["id"],
                "reason": "Automatic rollback",
            }
        }
        strategy = StateMappersStrategy(state_mappers)

        # Later, during compensation:
        params = strategy.extract(
            result={"booking": {"id": "ABC123"}},
            original_params={"dest": "NYC"},
            tool_name="book_flight",
        )
        # Returns: {"booking_id": "ABC123", "reason": "Automatic rollback"}
    """

    def __init__(
        self,
        state_mappers: dict[str, StateMapperFn] | None = None,
    ):
        """Initialize with mapping functions.

        Args:
            state_mappers: Dict mapping tool name -> extraction function.
                Function signature: (result, original_params) -> Dict[str, Any]
        """
        self.state_mappers = state_mappers or {}

    def extract(
        self,
        result: Any,
        original_params: dict[str, Any],
        compensation_tool: ToolLike | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Extract parameters using developer-provided mapper.

        Args:
            result: The result from the original tool call
            original_params: The original parameters passed to the tool
            compensation_tool: The compensation tool (unused)
            tool_name: Name of the original tool (for lookup)

        Returns:
            Extracted parameters dict, or None if no mapper exists
        """
        if tool_name and tool_name in self.state_mappers:
            try:
                return self.state_mappers[tool_name](result, original_params)
            except Exception:
                # Mapper failed, defer to next strategy
                return None
        return None

    def add_mapper(self, tool_name: str, mapper: StateMapperFn) -> None:
        """Add or replace a state mapper for a tool.

        Args:
            tool_name: Name of the tool
            mapper: The mapping function
        """
        self.state_mappers[tool_name] = mapper

    def remove_mapper(self, tool_name: str) -> bool:
        """Remove a state mapper for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            True if mapper was removed, False if it didn't exist
        """
        if tool_name in self.state_mappers:
            del self.state_mappers[tool_name]
            return True
        return False
