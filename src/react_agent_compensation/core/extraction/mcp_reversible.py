"""Extraction strategy for MCP reversible updates.

Extracts previous_* fields from update results to enable self-compensation
for tools annotated with x-reversible: true.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from react_agent_compensation.core.extraction.base import ExtractionStrategy, ToolLike

if TYPE_CHECKING:
    from react_agent_compensation.core.mcp.metadata import MCPToolMetadata


# common identifying parameter names to preserve from original call
IDENTIFYING_PARAMS = frozenset({
    "name", "id", "task_id", "pickup_id", "member_id",
    "user_id", "item_id", "record_id", "entity_id",
})


class MCPReversibleExtractionStrategy(ExtractionStrategy):
    """Extract previous_* fields for self-compensating updates.

    For tools marked with x-reversible: true and x-action-type: update,
    this strategy extracts previous state from the result to enable
    calling the same tool with previous values for compensation.

    Example:
        Result: {"message": "Updated", "previous_status": "pending", "previous_location": "NYC"}
        Original params: {"name": "John", "status": "completed"}

        Extracted: {"name": "John", "status": "pending", "location": "NYC"}
    """

    def __init__(self, tool_metadata: dict[str, "MCPToolMetadata"]):
        """Initialize with tool metadata mapping.

        Args:
            tool_metadata: Dict mapping tool names to MCPToolMetadata
        """
        self._metadata = tool_metadata

    def extract(
        self,
        result: Any,
        original_params: dict[str, Any],
        compensation_tool: ToolLike | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Extract compensation parameters from reversible update result.

        Only processes tools that are:
        1. Present in metadata
        2. Marked as reversible (x-reversible: true)
        3. Of action type "update"

        Args:
            result: The result from the original tool call
            original_params: The original parameters passed to the tool
            compensation_tool: Not used (compensator is self)
            tool_name: Name of the original tool

        Returns:
            Dict of parameters for calling same tool with previous values,
            or None if not applicable
        """
        if not tool_name:
            return None

        # check if this tool is a reversible update
        metadata = self._metadata.get(tool_name)
        if not metadata:
            return None

        if not metadata.is_reversible:
            return None

        if metadata.action_type != "update":
            return None

        # extract previous_* fields from result
        params = self._extract_previous_fields(result)

        # include identifying params from original call
        self._include_identifying_params(params, original_params)

        return params if params else None

    def _extract_previous_fields(self, result: Any) -> dict[str, Any]:
        """Extract all previous_* fields from result.

        Strips the "previous_" prefix to get the actual parameter name.

        Args:
            result: Tool result (should be dict)

        Returns:
            Dict mapping param names to previous values
        """
        params: dict[str, Any] = {}

        if not isinstance(result, dict):
            return params

        for key, value in result.items():
            if key.startswith("previous_") and value is not None:
                # strip "previous_" prefix (9 chars)
                param_name = key[9:]
                params[param_name] = value

        return params

    def _include_identifying_params(
        self,
        params: dict[str, Any],
        original_params: dict[str, Any],
    ) -> None:
        """Add identifying parameters from original call.

        These are needed to target the correct record when calling
        the same tool for compensation.

        Args:
            params: Dict to update with identifying params
            original_params: Original parameters from the call
        """
        for key, value in original_params.items():
            # include known identifying params
            if key in IDENTIFYING_PARAMS:
                params[key] = value
            # also include any *_id params not already present
            elif key.endswith("_id") and key not in params:
                params[key] = value

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "MCPReversibleExtractionStrategy"
