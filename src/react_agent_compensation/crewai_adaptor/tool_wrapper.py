"""Tool wrapping utilities for CrewAI compensation.

Provides functions to wrap CrewAI tools with compensation tracking
and automatic recovery/rollback behavior.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from react_agent_compensation.crewai_adaptor.middleware import CrewAICompensationMiddleware


logger = logging.getLogger(__name__)


def format_compensation_message(
    failed_action: str,
    error: str,
    recovery_attempts: int,
    compensated_actions: list[str],
    rollback_details: list[dict[str, Any]] | None = None,
    failure_context_summary: str = "",
    goals: list[str] | None = None,
) -> str:
    """Format an informative message for the LLM after compensation.

    Includes Strategic Context Preservation and Goal-Aware Recovery
    guidance for the LLM to make informed decisions.

    Args:
        failed_action: Name of the tool that failed
        error: Error message
        recovery_attempts: Number of retry attempts made
        compensated_actions: List of record IDs that were compensated
        rollback_details: Optional details about what was rolled back
        failure_context_summary: Cumulative failure context
        goals: Optional list of optimization goals

    Returns:
        Formatted message string for the LLM
    """
    lines = []

    # Strategic Context Preservation: Include cumulative failure context
    if failure_context_summary:
        lines.append(failure_context_summary)
        lines.append("")

    lines.extend([
        "[COMPENSATION TRIGGERED]",
        "",
        f"Action '{failed_action}' failed after {recovery_attempts} retry attempt(s).",
        f"Error: {error}",
        "",
    ])

    if compensated_actions or rollback_details:
        lines.append("[ROLLBACK EXECUTED - THESE ACTIONS WERE CANCELLED]")
        if rollback_details:
            for detail in rollback_details:
                action = detail.get('action', 'unknown')
                compensator = detail.get('compensator', 'unknown')
                params = detail.get('params', {})
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                lines.append(f"  - {action}({param_str}) -> CANCELLED via {compensator}")
        elif compensated_actions:
            for record_id in compensated_actions:
                lines.append(f"  - Action {record_id} CANCELLED")
        lines.append("")
        lines.append("IMPORTANT: The above actions were rolled back and need to be RE-DONE.")
        lines.append("")

    lines.append("State has been reset to before the failed sequence.")
    lines.append("")

    # Goal-Aware Recovery: Remind the LLM of optimization objectives
    if goals:
        lines.append("[REPLANNING GUIDANCE]")
        lines.append("You must now create a NEW complete plan that:")
        lines.append("  1. Re-schedules ALL the cancelled actions listed above")
        lines.append("  2. Avoids the failed approach (use different parameters/resources)")
        lines.append("  3. Optimizes for these goals:")
        for goal in goals:
            lines.append(f"       - {goal}")
        lines.append("")
        lines.append("Think holistically: plan ALL remaining work, not just the next step.")
    else:
        lines.append("You can now try a different approach or alternative parameters.")

    return "\n".join(lines)


def wrap_tool_with_compensation(
    tool: Any,
    middleware: "CrewAICompensationMiddleware",
    auto_rollback: bool = True,
    auto_recover: bool = True,
) -> Any:
    """Wrap a CrewAI tool with compensation tracking and recovery.

    This creates a new tool that:
    1. Records the action before execution
    2. On failure: attempts recovery (retry + alternatives)
    3. If recovery fails: triggers rollback of all previous actions
    4. Returns informative message to LLM on failure

    Args:
        tool: CrewAI tool (@tool function or BaseTool)
        middleware: CrewAICompensationMiddleware instance
        auto_rollback: Whether to automatically rollback on unrecoverable failure
        auto_recover: Whether to automatically attempt recovery

    Returns:
        Wrapped tool with same interface but compensation tracking
    """
    tool_name = _get_tool_name(tool)
    is_compensatable = middleware.is_compensatable(tool_name)

    # Get the original function/method
    original_func = _get_tool_func(tool)
    if not original_func:
        logger.warning(f"Cannot wrap tool {tool_name}: no callable found")
        return tool

    @functools.wraps(original_func)
    def wrapped_func(**kwargs) -> str:
        """Wrapped tool function with compensation tracking."""
        params = kwargs.copy()
        record = None

        # Only track compensatable tools
        if is_compensatable:
            record = middleware.rc_manager.record_action(tool_name, params)
            logger.debug(f"[COMPENSATION] Recorded action: {tool_name} (id={record.id})")

        def _handle_failure(error_msg: str, record_id: str) -> str:
            """Handle tool failure with recovery and rollback."""
            recovery_attempts = 0
            compensated_actions = []
            rollback_details = []
            failure_context_summary = ""

            # Step 1: Try recovery (retry + alternatives)
            if auto_recover:
                logger.info(f"[COMPENSATION] Attempting recovery for {tool_name}...")
                try:
                    recovery_result = middleware.rc_manager.recover(record_id, error_msg)
                    recovery_attempts = recovery_result.attempts

                    if recovery_result.success:
                        logger.info(
                            f"[COMPENSATION] Recovery succeeded for {tool_name} "
                            f"via {recovery_result.action_taken}"
                        )
                        # Convert result to string for CrewAI
                        result = recovery_result.result
                        return str(result) if not isinstance(result, str) else result

                    logger.warning(
                        f"[COMPENSATION] Recovery failed for {tool_name} "
                        f"after {recovery_attempts} attempt(s)"
                    )
                except Exception as recovery_error:
                    logger.error(f"[COMPENSATION] Recovery error: {recovery_error}")

            # Get Strategic Context Preservation summary
            try:
                failure_context_summary = middleware.get_failure_summary()
            except Exception as ctx_error:
                logger.debug(f"[COMPENSATION] Could not get failure context: {ctx_error}")

            # Step 2: Recovery failed - trigger rollback
            if auto_rollback:
                logger.info(f"[COMPENSATION] Triggering rollback for {tool_name}")
                try:
                    # Get rollback plan for details
                    rollback_plan = middleware.transaction_log.get_rollback_plan()
                    for rec in rollback_plan:
                        rollback_details.append({
                            'action': rec.action,
                            'compensator': rec.compensator,
                            'params': rec.params,
                        })

                    # Execute rollback
                    rollback_result = middleware.rollback()
                    compensated_actions = getattr(rollback_result, 'compensated', [])
                    msg = getattr(rollback_result, 'message', 'completed')
                    logger.info(f"[COMPENSATION] Rollback completed: {msg}")
                except Exception as rollback_error:
                    logger.error(f"[COMPENSATION] Rollback failed: {rollback_error}")

            # Step 3: Return informative message with Strategic Context
            return format_compensation_message(
                failed_action=tool_name,
                error=error_msg,
                recovery_attempts=recovery_attempts,
                compensated_actions=compensated_actions,
                rollback_details=rollback_details,
                failure_context_summary=failure_context_summary,
                goals=middleware.goals,
            )

        try:
            # Execute the original tool
            result = _execute_tool(tool, kwargs)

            # Check if result indicates an error
            is_error, error_msg = _detect_error(result)

            if is_error and record:
                middleware.rc_manager.mark_failed(record.id, error_msg)
                logger.warning(f"[COMPENSATION] Tool {tool_name} returned error: {error_msg}")
                return _handle_failure(error_msg, record.id)

            # Success - mark completed
            if record:
                middleware.rc_manager.mark_completed(record.id, result)
                logger.debug(f"[COMPENSATION] Marked completed: {tool_name}")

            # Ensure string return for CrewAI
            return str(result) if not isinstance(result, str) else result

        except Exception as e:
            error_msg = str(e)
            if record:
                middleware.rc_manager.mark_failed(record.id, error_msg)
                logger.error(f"[COMPENSATION] Tool {tool_name} failed: {e}")
                return _handle_failure(error_msg, record.id)
            raise

    # Create wrapped tool preserving CrewAI interface
    return _create_wrapped_tool(tool, wrapped_func)


def _get_tool_name(tool: Any) -> str:
    """Extract tool name from various tool types."""
    if hasattr(tool, "name"):
        return tool.name
    if callable(tool) and hasattr(tool, "__name__"):
        return tool.__name__
    return str(tool)


def _get_tool_func(tool: Any) -> Any:
    """Get the underlying function from a tool."""
    if hasattr(tool, "func") and callable(tool.func):
        return tool.func
    if hasattr(tool, "_run"):
        return tool._run
    if callable(tool):
        return tool
    return None


def _execute_tool(tool: Any, kwargs: dict[str, Any]) -> Any:
    """Execute a CrewAI tool."""
    if hasattr(tool, "func") and callable(tool.func):
        return tool.func(**kwargs)
    if hasattr(tool, "_run"):
        return tool._run(**kwargs)
    if callable(tool):
        return tool(**kwargs)
    raise ValueError(f"Cannot execute tool: {tool}")


def _detect_error(result: Any) -> tuple[bool, str]:
    """Detect if a result indicates an error.

    Returns:
        Tuple of (is_error, error_message)
    """
    import json

    # Parse JSON if string
    parsed = result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            # Not valid JSON, keep original string value for further checks
            pass

    # Check dict for error field
    if isinstance(parsed, dict):
        if parsed.get("error"):
            return True, str(parsed.get("error"))
        if parsed.get("status") == "failed":
            return True, str(parsed.get("message", "Operation failed"))

    # Check string for explicit error patterns to avoid false positives
    if isinstance(result, str):
        text = result.strip().lower()
        error_prefixes = ("error:", "error ", "failed:", "failed ", "failure:", "exception:")
        if text.startswith(error_prefixes):
            return True, result

    return False, ""


def _create_wrapped_tool(original_tool: Any, wrapped_func: Any) -> Any:
    """Create a wrapped tool preserving the CrewAI interface."""
    try:
        from crewai.tools import BaseTool, tool as tool_decorator
    except ImportError:
        # If CrewAI not installed, return function with attributes
        wrapped_func.name = _get_tool_name(original_tool)
        if hasattr(original_tool, "description"):
            wrapped_func.description = original_tool.description
        return wrapped_func

    # Check if original is a @tool decorated function
    if hasattr(original_tool, "name") and hasattr(original_tool, "description"):
        # Re-apply @tool decorator
        wrapped = tool_decorator(original_tool.name)(wrapped_func)
        wrapped.description = original_tool.description
        return wrapped

    # For BaseTool subclasses, create a wrapper class
    if isinstance(original_tool, BaseTool):
        return _create_base_tool_wrapper(original_tool, wrapped_func)

    # Default: add attributes to function
    wrapped_func.name = _get_tool_name(original_tool)
    if hasattr(original_tool, "description"):
        wrapped_func.description = original_tool.description
    return wrapped_func


def _create_base_tool_wrapper(original: Any, wrapped_func: Any) -> Any:
    """Create a wrapper for BaseTool subclasses."""
    try:
        from crewai.tools import BaseTool
        from pydantic import Field
    except ImportError:
        wrapped_func.name = original.name
        wrapped_func.description = original.description
        return wrapped_func

    class WrappedTool(BaseTool):
        name: str = original.name
        description: str = original.description

        def _run(self, **kwargs) -> str:
            return wrapped_func(**kwargs)

    return WrappedTool()
