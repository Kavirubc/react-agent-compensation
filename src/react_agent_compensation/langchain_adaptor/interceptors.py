"""LangChain-specific interceptors for tool calls and errors.

Provides interceptors that integrate the Core RecoveryManager
with LangChain's tool execution flow.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable

from react_agent_compensation.core.errors import create_error_detector
from react_agent_compensation.core.recovery_manager import RecoveryManager

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class ToolCallInterceptor:
    """Intercepts tool calls to provide recovery and compensation.

    Wraps tool execution with:
    - Action recording before execution
    - Error detection after execution
    - Recovery (retry + alternatives) on failure
    - Rollback trigger on unrecoverable failure

    Example:
        interceptor = ToolCallInterceptor(
            rc_manager=RecoveryManager(...),
            tools_cache={"my_tool": my_tool},
        )

        # In middleware:
        result = interceptor.intercept(
            tool_name="my_tool",
            params={"arg": "value"},
            tool_call_id="tc-123",
            execute_fn=lambda: tool.invoke(params),
        )
    """

    def __init__(
        self,
        rc_manager: RecoveryManager,
        tools_cache: dict[str, Any] | None = None,
        error_detector: Any | None = None,
    ):
        """Initialize interceptor.

        Args:
            rc_manager: Core's RecoveryManager
            tools_cache: Dict mapping tool names to tool instances
            error_detector: Error detection strategy (uses default if None)
        """
        self.rc_manager = rc_manager
        self._tools_cache = tools_cache or {}
        self._error_detector = error_detector or create_error_detector()

    def intercept(
        self,
        tool_name: str,
        params: dict[str, Any],
        tool_call_id: str,
        execute_fn: Callable[[], Any],
    ) -> InterceptResult:
        """Intercept a tool call with recovery handling.

        Args:
            tool_name: Name of the tool being called
            params: Parameters for the tool
            tool_call_id: Unique ID for this call
            execute_fn: Function to execute the tool

        Returns:
            InterceptResult with execution result or error info
        """
        record = None
        is_compensatable = self.rc_manager.is_compensatable(tool_name)

        # Record action before execution if compensatable
        if is_compensatable:
            record = self.rc_manager.record_action(tool_name, params)
            logger.debug(f"Recorded compensatable action: {tool_name} ({record.id})")

        # Execute the tool
        try:
            result = execute_fn()

            # Check for error in result
            if self._is_error(result):
                error_msg = self._get_error_message(result)
                logger.warning(f"Tool {tool_name} returned error: {error_msg}")

                if record:
                    # Try recovery
                    recovery = self.rc_manager.recover(
                        record.id,
                        error_msg or "Unknown error",
                        execute_fn=lambda name, args: self._execute_tool(name, args),
                    )
                    if recovery.success:
                        return InterceptResult(
                            success=True,
                            result=recovery.result,
                            recovered=True,
                            action_taken=recovery.action_taken,
                        )

                    # Recovery failed - trigger rollback
                    rollback = self.rc_manager.rollback(record.id)
                    return InterceptResult(
                        success=False,
                        error=error_msg,
                        rolled_back=True,
                        rollback_message=rollback.message,
                    )

                return InterceptResult(success=False, error=error_msg)

            # Success - mark completed
            if record:
                self.rc_manager.mark_completed(record.id, result)
                logger.debug(f"Marked action completed: {record.id}")

            return InterceptResult(success=True, result=result)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Tool {tool_name} raised exception: {error_msg}")

            if record:
                self.rc_manager.mark_failed(record.id, error_msg)

                # Try recovery
                recovery = self.rc_manager.recover(
                    record.id,
                    e,
                    execute_fn=lambda name, args: self._execute_tool(name, args),
                )
                if recovery.success:
                    return InterceptResult(
                        success=True,
                        result=recovery.result,
                        recovered=True,
                        action_taken=recovery.action_taken,
                    )

                # Recovery failed - trigger rollback
                rollback = self.rc_manager.rollback(record.id)
                return InterceptResult(
                    success=False,
                    error=error_msg,
                    rolled_back=True,
                    rollback_message=rollback.message,
                )

            return InterceptResult(success=False, error=error_msg, exception=e)

    def _execute_tool(self, name: str, params: dict[str, Any]) -> Any:
        """Execute a tool by name."""
        tool = self._tools_cache.get(name)
        if tool:
            return tool.invoke(params)
        raise ValueError(f"Tool {name} not found")

    def _is_error(self, result: Any) -> bool:
        """Check if result indicates an error."""
        return self._error_detector.is_error(result) or False

    def _get_error_message(self, result: Any) -> str | None:
        """Extract error message from result."""
        return self._error_detector.get_error_message(result)


class InterceptResult:
    """Result of an intercepted tool call."""

    def __init__(
        self,
        success: bool,
        result: Any = None,
        error: str | None = None,
        exception: Exception | None = None,
        recovered: bool = False,
        action_taken: str = "",
        rolled_back: bool = False,
        rollback_message: str = "",
    ):
        """Initialize intercept result.

        Args:
            success: Whether the call succeeded
            result: Result value if successful
            error: Error message if failed
            exception: Exception if one was raised
            recovered: Whether recovery was successful
            action_taken: Recovery action taken ("retry", "alternative")
            rolled_back: Whether rollback was performed
            rollback_message: Message from rollback operation
        """
        self.success = success
        self.result = result
        self.error = error
        self.exception = exception
        self.recovered = recovered
        self.action_taken = action_taken
        self.rolled_back = rolled_back
        self.rollback_message = rollback_message

    def to_tool_message(self, tool_call_id: str, tool_name: str) -> Any:
        """Convert to LangChain ToolMessage.

        Args:
            tool_call_id: Tool call ID
            tool_name: Tool name

        Returns:
            ToolMessage instance
        """
        try:
            from langchain_core.messages import ToolMessage

            if self.success:
                return ToolMessage(
                    content=str(self.result),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            else:
                content = self._build_error_content()
                return ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="error",
                )
        except ImportError:
            return self.result if self.success else self.error

    def _build_error_content(self) -> str:
        """Build error message content."""
        parts = []

        if self.rolled_back:
            parts.append("[ROLLBACK COMPLETE]")
            if self.rollback_message:
                parts.append(self.rollback_message)
        elif self.recovered:
            parts.append(f"[RECOVERED via {self.action_taken}]")

        if self.error:
            parts.append(f"Error: {self.error}")

        return "\n".join(parts)
