"""Base classes for extraction strategies.

This module defines:
- ExtractionStrategy: Abstract base class for all extraction strategies
- CompensationSchema: Declarative schema for parameter extraction
- ToolLike: Protocol for tool objects
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from react_agent_compensation.core.extraction.path_resolver import resolve_path


@runtime_checkable
class ToolLike(Protocol):
    """Protocol for tool objects - framework agnostic."""

    @property
    def name(self) -> str:
        """Name of the tool."""
        ...

    def get_input_schema(self) -> dict[str, Any]:
        """Return JSON schema for tool parameters."""
        ...


class CompensationSchema(BaseModel):
    """Declarative schema for compensation parameter extraction.

    Allows developers to explicitly declare how to map result fields
    to compensation tool parameters without writing a full state_mapper.

    Path Syntax:
        - "result.field" - Access result["field"]
        - "result.nested.field" - Access result["nested"]["field"]
        - "params.field" - Access original params["field"]
        - "result[0]" - Access result[0] (list index)
        - "result.items[0].id" - Nested with list access
        - "result.optional_field?" - Optional (won't error if missing)

    Example:
        schema = CompensationSchema(
            param_mapping={
                "booking_id": "result.id",
                "confirmation": "result.conf_code",
                "origin": "params.origin",
            },
            static_params={"reason": "Auto rollback"}
        )
    """

    param_mapping: dict[str, str] = Field(default_factory=dict)
    """Maps compensation param name -> path expression to extract value."""

    static_params: dict[str, Any] = Field(default_factory=dict)
    """Static parameters to always include (e.g., {"reason": "Auto rollback"})."""

    def extract(self, result: Any, original_params: dict[str, Any]) -> dict[str, Any]:
        """Extract compensation parameters using the schema.

        Args:
            result: The result from the original tool call
            original_params: The original parameters passed to the tool

        Returns:
            Dictionary of extracted parameters

        Raises:
            ValueError: If required path cannot be resolved
        """
        from react_agent_compensation.core.extraction.path_resolver import resolve_path

        extracted = dict(self.static_params)
        context = {"result": result, "params": original_params}

        for param_name, path_expr in self.param_mapping.items():
            optional = path_expr.endswith("?")
            if optional:
                path_expr = path_expr[:-1]

            try:
                value = resolve_path(path_expr, context)
                extracted[param_name] = value
            except (KeyError, IndexError, TypeError) as e:
                if not optional:
                    raise ValueError(
                        f"Cannot extract '{param_name}' from path '{path_expr}': {e}"
                    ) from e

        return extracted


class ExtractionStrategy(ABC):
    """Abstract base class for parameter extraction strategies.

    Each strategy attempts to extract compensation parameters from a
    tool result. Strategies can:
    - Return dict: Successfully extracted parameters
    - Return None: Cannot extract, defer to next strategy

    This allows strategies to be chained in priority order.
    """

    @abstractmethod
    def extract(
        self,
        result: Any,
        original_params: dict[str, Any],
        compensation_tool: ToolLike | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Attempt to extract compensation parameters.

        Args:
            result: The result from the original tool call
            original_params: The original parameters passed to the tool
            compensation_tool: The compensation tool (for schema inspection)
            tool_name: Name of the original tool (for lookup)

        Returns:
            Dict of extracted parameters, or None to defer to next strategy
        """
        pass

    @property
    def name(self) -> str:
        """Human-readable name for logging and debugging."""
        return self.__class__.__name__
