"""Type adapters implementing Core protocols for LangChain types.

This module provides adapters that bridge LangChain-specific types
to the framework-agnostic Core protocols.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from react_agent_compensation.core.protocols import ActionResult

if TYPE_CHECKING:
    pass


class LangChainActionResult:
    """Adapts LangChain ToolMessage to ActionResult protocol.

    Example:
        from langchain_core.messages import ToolMessage

        msg = ToolMessage(content="done", tool_call_id="123", name="my_tool")
        result = LangChainActionResult(msg)
        print(result.content)  # "done"
        print(result.status)   # None (or "error" if set)
    """

    def __init__(self, tool_message: Any):
        """Initialize with a ToolMessage.

        Args:
            tool_message: LangChain ToolMessage instance
        """
        self._msg = tool_message

    @property
    def content(self) -> Any:
        """Get message content."""
        return self._msg.content

    @property
    def status(self) -> str | None:
        """Get message status if available."""
        return getattr(self._msg, "status", None)

    @property
    def name(self) -> str:
        """Get tool name."""
        return getattr(self._msg, "name", "")

    @property
    def action_id(self) -> str:
        """Get tool call ID."""
        return getattr(self._msg, "tool_call_id", "")


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
            action_id: Tool call ID
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


class LangChainToolExecutor:
    """Adapts LangChain tools to ActionExecutor protocol.

    Example:
        executor = LangChainToolExecutor(tools_cache)
        result = executor.execute("my_tool", {"arg": "value"})
    """

    def __init__(self, tools_cache: dict[str, Any]):
        """Initialize with tools cache.

        Args:
            tools_cache: Dict mapping tool names to tool instances
        """
        self._tools = tools_cache

    def execute(self, name: str, params: dict[str, Any]) -> ActionResult:
        """Execute a tool by name.

        Args:
            name: Tool name
            params: Tool parameters

        Returns:
            ActionResult with execution result
        """
        tool = self._tools.get(name)
        if not tool:
            return SimpleActionResult(
                content=f"Tool {name} not found",
                status="error",
                name=name,
            )

        try:
            result = tool.invoke(params)
            return SimpleActionResult(
                content=result,
                status="success",
                name=name,
            )
        except Exception as e:
            return SimpleActionResult(
                content=f"Error: {e}",
                status="error",
                name=name,
            )


class LangChainToolSchema:
    """Adapts LangChain BaseTool to ToolSchemaProvider protocol."""

    def __init__(self, tool: Any):
        """Initialize with a LangChain tool.

        Args:
            tool: LangChain BaseTool instance
        """
        self._tool = tool

    @property
    def name(self) -> str:
        """Get tool name."""
        return self._tool.name

    @property
    def description(self) -> str:
        """Get tool description."""
        return getattr(self._tool, "description", "") or ""

    def get_input_schema(self) -> dict[str, Any]:
        """Get input schema as JSON Schema dict."""
        if hasattr(self._tool, "args_schema") and self._tool.args_schema:
            try:
                return self._tool.args_schema.schema()
            except Exception:
                pass
        return {}


def build_tools_cache(tools: list[Any] | None) -> dict[str, Any]:
    """Build tools cache from list of tools.

    Args:
        tools: List of LangChain tools

    Returns:
        Dict mapping tool names to tool instances
    """
    cache: dict[str, Any] = {}
    for tool in tools or []:
        if hasattr(tool, "name"):
            cache[tool.name] = tool
    return cache


def build_tool_schemas(tools: list[Any] | None) -> dict[str, LangChainToolSchema]:
    """Build schema adapters from tools list.

    Args:
        tools: List of LangChain tools

    Returns:
        Dict mapping tool names to schema adapters
    """
    schemas: dict[str, LangChainToolSchema] = {}
    for tool in tools or []:
        if hasattr(tool, "name"):
            schemas[tool.name] = LangChainToolSchema(tool)
    return schemas
