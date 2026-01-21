"""Type adapters implementing Core protocols for CrewAI types.

This module provides adapters that bridge CrewAI-specific types
to the framework-agnostic Core protocols.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable


class CrewAIActionResult:
    """Adapts CrewAI tool results to ActionResult protocol.

    CrewAI tools return strings, so we parse JSON when possible
    to extract structured data.

    Example:
        result = CrewAIActionResult('{"booking_id": "123"}', "book_flight")
        print(result.content)  # {"booking_id": "123"}
        print(result.name)     # "book_flight"
    """

    def __init__(
        self,
        result: str,
        tool_name: str,
        status: str | None = None,
        action_id: str | None = None,
    ):
        """Initialize with a CrewAI tool result.

        Args:
            result: String result from CrewAI tool
            tool_name: Name of the tool that produced this result
            status: Optional status indicator
            action_id: Optional unique identifier for this action
        """
        self._raw = result
        self._name = tool_name
        self._status = status
        self._action_id = action_id or str(uuid.uuid4())
        self._parsed = self._parse_result(result)

    def _parse_result(self, result: str) -> Any:
        """Parse string result to dict if JSON, otherwise return as-is."""
        if not isinstance(result, str):
            return result

        # Try to parse as JSON
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result

    @property
    def content(self) -> Any:
        """Get parsed content."""
        return self._parsed

    @property
    def raw(self) -> str:
        """Get raw string result."""
        return self._raw

    @property
    def status(self) -> str | None:
        """Get status if available."""
        if self._status:
            return self._status

        # Detect error from content
        if self._is_error_content():
            return "error"
        return None

    @property
    def name(self) -> str:
        """Get tool name."""
        return self._name

    @property
    def action_id(self) -> str:
        """Get action identifier."""
        return self._action_id

    def _is_error_content(self) -> bool:
        """Check if content indicates an error."""
        if isinstance(self._parsed, dict):
            return bool(self._parsed.get("error"))

        if isinstance(self._raw, str):
            error_indicators = ["error", "failed", "failure", "exception"]
            return any(ind in self._raw.lower() for ind in error_indicators)

        return False


class CrewAIToolExecutor:
    """Adapts CrewAI tools to ActionExecutor protocol.

    Handles both @tool decorated functions and BaseTool subclasses.

    Example:
        executor = CrewAIToolExecutor(tools_cache)
        result = executor.execute("book_flight", {"destination": "NYC"})
    """

    def __init__(self, tools_cache: dict[str, Any]):
        """Initialize with tools cache.

        Args:
            tools_cache: Dict mapping tool names to tool instances/functions
        """
        self._tools = tools_cache

    def execute(self, name: str, params: dict[str, Any]) -> CrewAIActionResult:
        """Execute a tool by name.

        Args:
            name: Tool name
            params: Tool parameters

        Returns:
            CrewAIActionResult with execution result
        """
        tool = self._tools.get(name)
        if not tool:
            return CrewAIActionResult(
                result=f"Tool {name} not found",
                tool_name=name,
                status="error",
            )

        try:
            result = self._invoke_tool(tool, params)
            return CrewAIActionResult(
                result=str(result) if not isinstance(result, str) else result,
                tool_name=name,
            )
        except Exception as e:
            return CrewAIActionResult(
                result=f"Error: {e}",
                tool_name=name,
                status="error",
            )

    def _invoke_tool(self, tool: Any, params: dict[str, Any]) -> Any:
        """Invoke a CrewAI tool with parameters.

        Handles:
        - @tool decorated functions (have func attribute)
        - BaseTool subclasses (have _run method)
        - Plain callables
        """
        # @tool decorated function
        if hasattr(tool, "func") and callable(tool.func):
            return tool.func(**params)

        # BaseTool with _run method
        if hasattr(tool, "_run"):
            return tool._run(**params)

        # Plain callable
        if callable(tool):
            return tool(**params)

        raise ValueError(f"Cannot invoke tool: {tool}")


class SimpleActionResult:
    """Simple ActionResult implementation for internal use."""

    def __init__(
        self,
        content: Any,
        status: str | None = None,
        name: str = "",
        action_id: str | None = None,
    ):
        """Initialize simple result.

        Args:
            content: Result content
            status: Status string (e.g., "error", "success")
            name: Tool name
            action_id: Unique action identifier
        """
        self._content = content
        self._status = status
        self._name = name
        self._action_id = action_id or str(uuid.uuid4())

    @property
    def content(self) -> Any:
        return self._content

    @property
    def status(self) -> str | None:
        return self._status

    @property
    def name(self) -> str:
        return self._name

    @property
    def action_id(self) -> str:
        return self._action_id


def build_tools_cache(tools: list[Any] | None) -> dict[str, Any]:
    """Build tools cache from list of tools.

    Handles:
    - @tool decorated functions (have name attribute)
    - BaseTool subclasses (have name attribute)
    - Named callables

    Args:
        tools: List of CrewAI tools

    Returns:
        Dict mapping tool names to tool instances
    """
    cache: dict[str, Any] = {}
    for tool in tools or []:
        name = _get_tool_name(tool)
        if name:
            cache[name] = tool
    return cache


def _get_tool_name(tool: Any) -> str | None:
    """Extract tool name from various tool types."""
    # Standard name attribute
    if hasattr(tool, "name"):
        return tool.name

    # Function name
    if callable(tool) and hasattr(tool, "__name__"):
        return tool.__name__

    return None
