"""Strategy that passes through the entire result.

Last resort strategy - if no other extraction works, pass the
entire result to the compensation tool. This works when the
compensation tool accepts the same structure as the result.
"""

from __future__ import annotations

from typing import Any

from react_agent_compensation.core.extraction.base import ExtractionStrategy, ToolLike


class PassthroughStrategy(ExtractionStrategy):
    """Pass through the entire result as compensation params.

    Last resort strategy - if no other extraction works, pass the
    entire result to the compensation tool. This works when the
    compensation tool accepts the same structure as the result.

    Example:
        result = {"booking_id": "123", "amount": 500}
        # Returns: {"booking_id": "123", "amount": 500}

        result = "simple_string"
        # Returns: None (can't pass non-dict)
    """

    def __init__(self, allow_non_dict: bool = False):
        """Initialize passthrough strategy.

        Args:
            allow_non_dict: If True, wrap non-dict results in {"value": result}
        """
        self.allow_non_dict = allow_non_dict

    def extract(
        self,
        result: Any,
        original_params: dict[str, Any],
        compensation_tool: ToolLike | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Pass through the entire result as parameters.

        Args:
            result: The result from the original tool call
            original_params: The original parameters passed to the tool
            compensation_tool: The compensation tool (unused)
            tool_name: Name of the original tool (unused)

        Returns:
            Copy of result dict, or None if not a dict
        """
        if isinstance(result, dict):
            return dict(result)

        if self.allow_non_dict and result is not None:
            return {"value": result}

        return None
