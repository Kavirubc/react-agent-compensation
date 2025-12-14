"""Compensated MCP tool wrapper.

Wraps MCP tools with automatic compensation tracking - records actions
in TransactionLog and handles error detection with rollback triggering.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from react_agent_compensation.core.recovery_manager import RecoveryManager

logger = logging.getLogger(__name__)


class CompensatedMCPTool:
    """Wraps an MCP tool with compensation tracking.

    Intercepts tool calls to:
    1. Record the action in TransactionLog before execution
    2. Execute the underlying MCP tool
    3. Check result for errors
    4. Mark action as completed or failed

    Example:
        wrapped = CompensatedMCPTool(
            inner_tool=mcp_tool,
            recovery_manager=manager,
            metadata={"x-action-type": "create"},
        )
        result = wrapped.invoke({"name": "John"})
    """

    def __init__(
        self,
        inner_tool: Any,
        recovery_manager: "RecoveryManager",
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize the wrapped tool.

        Args:
            inner_tool: The underlying MCP tool (LangChain BaseTool)
            recovery_manager: RecoveryManager for tracking actions
            metadata: Tool metadata (x-action-type, x-reversible, etc.)
        """
        self._inner = inner_tool
        self._manager = recovery_manager
        self._metadata = metadata or {}

    @property
    def name(self) -> str:
        """Get tool name."""
        return self._inner.name

    @property
    def description(self) -> str:
        """Get tool description."""
        return getattr(self._inner, "description", "")

    @property
    def args_schema(self) -> Any:
        """Get tool args schema."""
        return getattr(self._inner, "args_schema", None)

    @property
    def metadata(self) -> dict[str, Any]:
        """Get tool metadata."""
        return self._metadata

    @property
    def is_compensatable(self) -> bool:
        """Check if this tool has a compensation pair."""
        return self._manager.is_compensatable(self.name)

    def invoke(self, input: dict[str, Any], config: dict | None = None) -> str:
        """Invoke the tool with compensation tracking.

        Args:
            input: Tool input parameters
            config: Optional LangChain config

        Returns:
            Tool result as string
        """
        record = None

        # Only record if compensatable
        if self.is_compensatable:
            record = self._manager.record_action(self.name, input)
            logger.debug(f"Recorded compensatable action: {self.name} (id={record.id})")

        try:
            # Execute the underlying tool
            result = self._inner.invoke(input, config=config)

            # Check for error in result
            if self._is_error_result(result):
                error_msg = self._extract_error_message(result)
                if record:
                    self._manager.mark_failed(record.id, error_msg)
                raise MCPToolError(error_msg, tool_name=self.name, result=result)

            # Mark completed
            if record:
                parsed_result = self._parse_result(result)
                self._manager.mark_completed(record.id, parsed_result)
                logger.debug(f"Marked action completed: {self.name}")

            return result

        except MCPToolError:
            raise
        except Exception as e:
            if record:
                self._manager.mark_failed(record.id, str(e))
            raise

    async def ainvoke(self, input: dict[str, Any], config: dict | None = None) -> str:
        """Async invoke the tool with compensation tracking.

        Args:
            input: Tool input parameters
            config: Optional LangChain config

        Returns:
            Tool result as string
        """
        record = None

        if self.is_compensatable:
            record = self._manager.record_action(self.name, input)
            logger.debug(f"Recorded compensatable action: {self.name} (id={record.id})")

        try:
            # Execute the underlying tool (async)
            result = await self._inner.ainvoke(input, config=config)

            if self._is_error_result(result):
                error_msg = self._extract_error_message(result)
                if record:
                    self._manager.mark_failed(record.id, error_msg)
                raise MCPToolError(error_msg, tool_name=self.name, result=result)

            if record:
                parsed_result = self._parse_result(result)
                self._manager.mark_completed(record.id, parsed_result)

            return result

        except MCPToolError:
            raise
        except Exception as e:
            if record:
                self._manager.mark_failed(record.id, str(e))
            raise

    def _is_error_result(self, result: Any) -> bool:
        """Check if the result indicates an error.

        Looks for common error patterns:
        - {"error": "..."} or {"status": "error"}
        - String containing "error"
        """
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                return self._check_error_in_dict(parsed)
            except (json.JSONDecodeError, TypeError):
                return False

        if isinstance(result, dict):
            return self._check_error_in_dict(result)

        return False

    def _check_error_in_dict(self, data: dict) -> bool:
        """Check if dict indicates an error."""
        if "error" in data:
            return True
        if data.get("status") == "error":
            return True
        return False

    def _extract_error_message(self, result: Any) -> str:
        """Extract error message from result."""
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                return parsed.get("error", str(result))
            except (json.JSONDecodeError, TypeError):
                return str(result)

        if isinstance(result, dict):
            return result.get("error", str(result))

        return str(result)

    def _parse_result(self, result: Any) -> dict[str, Any]:
        """Parse result into dict for storage."""
        if isinstance(result, dict):
            return result

        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return {"raw": result}

        return {"raw": str(result)}


class MCPToolError(Exception):
    """Error from MCP tool execution."""

    def __init__(self, message: str, tool_name: str, result: Any = None):
        super().__init__(message)
        self.tool_name = tool_name
        self.result = result


def wrap_mcp_tools(
    tools: list[Any],
    recovery_manager: "RecoveryManager",
    tool_metadata: dict[str, dict[str, Any]] | None = None,
) -> list[CompensatedMCPTool]:
    """Wrap multiple MCP tools with compensation tracking.

    Args:
        tools: List of MCP tools (LangChain BaseTools)
        recovery_manager: RecoveryManager for tracking
        tool_metadata: Optional metadata per tool (keyed by tool name)

    Returns:
        List of wrapped tools
    """
    tool_metadata = tool_metadata or {}
    wrapped = []

    for tool in tools:
        name = tool.name
        metadata = tool_metadata.get(name, {})
        wrapped_tool = CompensatedMCPTool(
            inner_tool=tool,
            recovery_manager=recovery_manager,
            metadata=metadata,
        )
        wrapped.append(wrapped_tool)
        logger.debug(f"Wrapped MCP tool: {name}")

    return wrapped
