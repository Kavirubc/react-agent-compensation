"""Hook-based integration for CrewAI compensation.

Provides global hooks that can be registered with CrewAI's
@before_tool_call and @after_tool_call decorators.

This is a secondary integration approach. The primary approach
is tool wrapping (see tool_wrapper.py).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from react_agent_compensation.crewai_adaptor.tool_wrapper import format_compensation_message

if TYPE_CHECKING:
    from react_agent_compensation.crewai_adaptor.middleware import CrewAICompensationMiddleware


logger = logging.getLogger(__name__)


class CrewAIHookManager:
    """Manages compensation hooks for CrewAI.

    Provides before/after tool call hooks that integrate with CrewAI's
    hook system for compensation tracking.

    This is useful when you want global compensation handling across
    a crew without wrapping individual tools.

    Example:
        from crewai import Crew

        middleware = CrewAICompensationMiddleware(...)
        hook_manager = CrewAIHookManager(middleware)

        # Register hooks manually
        @before_tool_call
        def before_hook(context):
            hook_manager.before_tool_call(context)

        @after_tool_call
        def after_hook(context):
            hook_manager.after_tool_call(context)
    """

    def __init__(
        self,
        middleware: "CrewAICompensationMiddleware",
        crew_id: str | None = None,
        auto_rollback: bool = True,
        auto_recover: bool = True,
    ):
        """Initialize hook manager.

        Args:
            middleware: CrewAICompensationMiddleware instance
            crew_id: Optional crew ID for filtering (only handle tools from this crew)
            auto_rollback: Whether to auto-rollback on unrecoverable failure
            auto_recover: Whether to attempt recovery on failure
        """
        self._middleware = middleware
        self._crew_id = crew_id
        self._auto_rollback = auto_rollback
        self._auto_recover = auto_recover
        # Map tool_use_id -> record_id for tracking
        self._pending_records: dict[str, str] = {}

    def before_tool_call(self, context: Any) -> None:
        """Hook to call before tool execution.

        Records the action if the tool is compensatable.

        Args:
            context: CrewAI ToolCallHookContext containing:
                - tool_name: Name of the tool being called
                - tool_input: Dict of input parameters (mutable)
                - tool: The tool instance
                - agent: The agent executing the tool
                - task: The current task
                - crew: The crew instance
        """
        # Filter by crew if specified
        if self._crew_id and hasattr(context, 'crew'):
            crew_id = getattr(context.crew, 'id', None) or str(context.crew)
            if crew_id != self._crew_id:
                return

        tool_name = getattr(context, 'tool_name', None)
        tool_input = getattr(context, 'tool_input', {}) or {}

        if not tool_name:
            return

        # Record action if compensatable
        if self._middleware.is_compensatable(tool_name):
            record = self._middleware.rc_manager.record_action(tool_name, tool_input)
            # Store mapping for after_tool_call
            call_id = self._generate_call_id(context)
            self._pending_records[call_id] = record.id
            logger.debug(f"[HOOK] Recorded action: {tool_name} (id={record.id})")

    def after_tool_call(self, context: Any) -> None:
        """Hook to call after tool execution.

        Checks for errors and handles recovery/rollback.

        Args:
            context: CrewAI ToolCallHookContext containing:
                - tool_name: Name of the tool
                - tool_result: Result from the tool (mutable)
                - tool_input: Dict of input parameters
                - agent: The agent
                - task: The current task
                - crew: The crew instance
        """
        # Filter by crew if specified
        if self._crew_id and hasattr(context, 'crew'):
            crew_id = getattr(context.crew, 'id', None) or str(context.crew)
            if crew_id != self._crew_id:
                return

        tool_name = getattr(context, 'tool_name', None)
        tool_result = getattr(context, 'tool_result', None)

        if not tool_name:
            return

        # Get the record ID for this call
        call_id = self._generate_call_id(context)
        record_id = self._pending_records.pop(call_id, None)

        if not record_id:
            # Not a compensatable action or not tracked
            return

        # Check for error in result
        is_error, error_msg = self._detect_error(tool_result)

        if is_error:
            logger.warning(f"[HOOK] Tool {tool_name} returned error: {error_msg}")
            self._middleware.rc_manager.mark_failed(record_id, error_msg)

            # Handle recovery and rollback
            new_result = self._handle_failure(tool_name, error_msg, record_id)

            # Modify the context result if possible
            if hasattr(context, 'tool_result'):
                context.tool_result = new_result
        else:
            # Success - mark completed
            self._middleware.rc_manager.mark_completed(record_id, tool_result)
            logger.debug(f"[HOOK] Marked completed: {tool_name}")

    def _generate_call_id(self, context: Any) -> str:
        """Generate a unique ID for a tool call context."""
        tool_name = getattr(context, 'tool_name', '')
        agent = getattr(context, 'agent', None)
        agent_id = getattr(agent, 'id', str(agent)) if agent else ''
        return f"{agent_id}:{tool_name}:{id(context)}"

    def _detect_error(self, result: Any) -> tuple[bool, str]:
        """Detect if a result indicates an error."""
        import json

        if result is None:
            return False, ""

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

        # Check string for explicit error patterns to avoid false positives
        if isinstance(result, str):
            text = result.strip().lower()
            error_prefixes = ("error:", "error ", "failed:", "failed ", "failure:", "exception:")
            if text.startswith(error_prefixes):
                return True, result

        return False, ""

    def _handle_failure(self, tool_name: str, error_msg: str, record_id: str) -> str:
        """Handle tool failure with recovery and rollback."""
        recovery_attempts = 0
        compensated_actions = []
        rollback_details = []
        failure_context_summary = ""

        # Try recovery
        if self._auto_recover:
            logger.info(f"[HOOK] Attempting recovery for {tool_name}...")
            try:
                recovery_result = self._middleware.rc_manager.recover(record_id, error_msg)
                recovery_attempts = recovery_result.attempts

                if recovery_result.success:
                    logger.info(f"[HOOK] Recovery succeeded via {recovery_result.action_taken}")
                    result = recovery_result.result
                    return str(result) if not isinstance(result, str) else result

            except Exception as recovery_error:
                logger.error(f"[HOOK] Recovery error: {recovery_error}")

        # Get failure context (best-effort, ignore errors)
        try:
            failure_context_summary = self._middleware.get_failure_summary()
        except Exception as e:
            logger.debug(f"[HOOK] Could not get failure context: {e}")

        # Trigger rollback
        if self._auto_rollback:
            logger.info(f"[HOOK] Triggering rollback for {tool_name}")
            try:
                rollback_plan = self._middleware.transaction_log.get_rollback_plan()
                for rec in rollback_plan:
                    rollback_details.append({
                        'action': rec.action,
                        'compensator': rec.compensator,
                        'params': rec.params,
                    })

                rollback_result = self._middleware.rollback()
                compensated_actions = getattr(rollback_result, 'compensated', [])
            except Exception as rollback_error:
                logger.error(f"[HOOK] Rollback failed: {rollback_error}")

        # Return formatted message
        return format_compensation_message(
            failed_action=tool_name,
            error=error_msg,
            recovery_attempts=recovery_attempts,
            compensated_actions=compensated_actions,
            rollback_details=rollback_details,
            failure_context_summary=failure_context_summary,
            goals=self._middleware.goals,
        )


def create_compensation_hooks(
    middleware: "CrewAICompensationMiddleware",
    crew_id: str | None = None,
) -> tuple[Any, Any]:
    """Create before/after hook functions for CrewAI.

    Returns tuple of (before_hook, after_hook) functions that can be
    registered with CrewAI's hook system.

    Args:
        middleware: CrewAICompensationMiddleware instance
        crew_id: Optional crew ID for filtering

    Returns:
        Tuple of (before_tool_call_hook, after_tool_call_hook)

    Example:
        before_hook, after_hook = create_compensation_hooks(middleware)

        @before_tool_call
        def my_before_hook(context):
            before_hook(context)

        @after_tool_call
        def my_after_hook(context):
            after_hook(context)
    """
    manager = CrewAIHookManager(middleware, crew_id=crew_id)
    return manager.before_tool_call, manager.after_tool_call
