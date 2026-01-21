"""Type adapters implementing Core protocols for AWS Strands types.

This module provides adapters that bridge Strands-specific types
to the framework-agnostic Core protocols.
"""

from __future__ import annotations

import json
import uuid
from typing import Any


class StrandsActionResult:
    """Adapts Strands ToolResult to ActionResult protocol.

    Strands tools return results in format:
    {content: [{type, text}], status, toolUseId}

    Example:
        result = StrandsActionResult(strands_result, "reserve_inventory")
        print(result.content)  # Parsed content
        print(result.status)   # "success" or "error"
    """

    def __init__(
        self,
        result: dict[str, Any] | str | None,
        tool_name: str,
        action_id: str | None = None,
    ):
        """Initialize with a Strands tool result.

        Args:
            result: Strands ToolResult dict or string result
            tool_name: Name of the tool that produced this result
            action_id: Optional unique identifier for this action
        """
        self._raw = result
        self._name = tool_name
        self._action_id = action_id or str(uuid.uuid4())
        self._parsed = self._parse_result(result)

    def _parse_result(self, result: dict[str, Any] | str | None) -> Any:
        """Parse Strands result to extract content."""
        if result is None:
            return None

        # String result - try JSON parsing
        if isinstance(result, str):
            try:
                return json.loads(result)
            except (json.JSONDecodeError, TypeError):
                return result

        # Dict result - extract from Strands format
        if isinstance(result, dict):
            # Strands format: {content: [{type, text}], status, toolUseId}
            content = result.get("content", [])
            if isinstance(content, list) and content:
                # Extract text from first content item
                text = content[0].get("text", "")
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    return text

            # Direct content
            return result.get("content", result)

        return result

    @property
    def content(self) -> Any:
        """Get parsed content."""
        return self._parsed

    @property
    def raw(self) -> dict[str, Any] | str | None:
        """Get raw result."""
        return self._raw

    @property
    def status(self) -> str | None:
        """Get status from result."""
        if isinstance(self._raw, dict):
            status = self._raw.get("status")
            if status:
                return status

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

        if isinstance(self._parsed, str):
            error_indicators = ["error", "failed", "failure", "exception"]
            return any(ind in self._parsed.lower() for ind in error_indicators)

        return False


class StrandsToolExecutor:
    """Adapts Strands tools to ActionExecutor protocol.

    Handles @tool decorated functions in Strands SDK.

    Example:
        executor = StrandsToolExecutor(tools_cache)
        result = executor.execute("reserve_inventory", {"product_id": "123"})
    """

    def __init__(self, tools_cache: dict[str, Any]):
        """Initialize with tools cache.

        Args:
            tools_cache: Dict mapping tool names to tool instances/functions
        """
        self._tools = tools_cache

    def execute(self, name: str, params: dict[str, Any]) -> StrandsActionResult:
        """Execute a tool by name (sync).

        Args:
            name: Tool name
            params: Tool parameters

        Returns:
            StrandsActionResult with execution result
        """
        tool = self._tools.get(name)
        if not tool:
            return StrandsActionResult(
                result={"content": [{"type": "text", "text": f"Tool {name} not found"}], "status": "error"},
                tool_name=name,
            )

        try:
            result = self._invoke_tool(tool, params)
            return StrandsActionResult(
                result=result,
                tool_name=name,
            )
        except Exception as e:
            return StrandsActionResult(
                result={"content": [{"type": "text", "text": f"Error: {e}"}], "status": "error"},
                tool_name=name,
            )

    async def execute_async(self, name: str, params: dict[str, Any]) -> StrandsActionResult:
        """Execute a tool by name (async).

        Args:
            name: Tool name
            params: Tool parameters

        Returns:
            StrandsActionResult with execution result
        """
        tool = self._tools.get(name)
        if not tool:
            return StrandsActionResult(
                result={"content": [{"type": "text", "text": f"Tool {name} not found"}], "status": "error"},
                tool_name=name,
            )

        try:
            result = await self._invoke_tool_async(tool, params)
            return StrandsActionResult(
                result=result,
                tool_name=name,
            )
        except Exception as e:
            return StrandsActionResult(
                result={"content": [{"type": "text", "text": f"Error: {e}"}], "status": "error"},
                tool_name=name,
            )

    def _invoke_tool(self, tool: Any, params: dict[str, Any]) -> Any:
        """Invoke a Strands tool with parameters (sync)."""
        # @tool decorated function
        if hasattr(tool, "func") and callable(tool.func):
            return tool.func(**params)

        # Direct callable
        if callable(tool):
            return tool(**params)

        raise ValueError(f"Cannot invoke tool: {tool}")

    async def _invoke_tool_async(self, tool: Any, params: dict[str, Any]) -> Any:
        """Invoke a Strands tool with parameters (async)."""
        import asyncio

        # Async function
        if hasattr(tool, "func") and callable(tool.func):
            func = tool.func
        elif callable(tool):
            func = tool
        else:
            raise ValueError(f"Cannot invoke tool: {tool}")

        # Check if async
        if asyncio.iscoroutinefunction(func):
            return await func(**params)
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(**params))


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
            status: Status string
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

    Handles Strands @tool decorated functions.

    Args:
        tools: List of Strands tools

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
