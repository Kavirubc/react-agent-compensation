"""HookProvider implementation for AWS Strands compensation.

Provides CompensationHookProvider that implements Strands' HookProvider
protocol for automatic compensation tracking and recovery.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable

from react_agent_compensation.core.config import AlternativeMap, CompensationPairs, RetryPolicy
from react_agent_compensation.core.extraction import CompensationSchema, create_extraction_strategy
from react_agent_compensation.core.recovery_manager import RecoveryManager
from react_agent_compensation.core.transaction_log import TransactionLog
from react_agent_compensation.strands_adaptor.adapters import StrandsToolExecutor, build_tools_cache
from react_agent_compensation.strands_adaptor.state import StrandsStateSync

if TYPE_CHECKING:
    pass


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

    Args:
        failed_action: Name of the tool that failed
        error: Error message
        recovery_attempts: Number of retry attempts made
        compensated_actions: List of record IDs that were compensated
        rollback_details: Details about what was rolled back
        failure_context_summary: Cumulative failure context
        goals: Optional list of optimization goals

    Returns:
        Formatted message string for the LLM
    """
    lines = []

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


class CompensationHookProvider:
    """HookProvider for compensation tracking in AWS Strands.

    Implements Strands' HookProvider protocol to automatically track
    tool executions and provide recovery/rollback on failures.

    This is the primary integration approach for Strands.

    Example:
        from strands import Agent

        provider = CompensationHookProvider(
            compensation_pairs={"reserve_inventory": "release_inventory"},
            tools=[reserve_inventory, release_inventory],
        )

        agent = Agent(
            system_prompt="...",
            tools=[reserve_inventory, release_inventory],
            hooks=[provider],
        )
    """

    def __init__(
        self,
        compensation_pairs: CompensationPairs,
        tools: list[Any] | None = None,
        *,
        alternative_map: AlternativeMap | None = None,
        retry_policy: RetryPolicy | None = None,
        compensation_schemas: dict[str, CompensationSchema] | None = None,
        state_mappers: dict[str, Callable] | None = None,
        goals: list[str] | None = None,
        auto_rollback: bool = True,
        auto_recover: bool = True,
        persist_state: bool = True,
    ):
        """Initialize hook provider.

        Args:
            compensation_pairs: Maps tool names to compensation tools
            tools: List of Strands tools
            alternative_map: Maps tools to alternatives to try on failure
            retry_policy: Configuration for retry behavior
            compensation_schemas: Declarative extraction schemas
            state_mappers: Custom extraction functions
            goals: Optimization goals for goal-aware recovery
            auto_rollback: Whether to automatically rollback on unrecoverable failure
            auto_recover: Whether to automatically attempt recovery
            persist_state: Whether to persist state to invocation_state
        """
        self._compensation_pairs = compensation_pairs
        self._alternative_map = alternative_map or {}
        self._goals = goals or []
        self._auto_rollback = auto_rollback
        self._auto_recover = auto_recover
        self._persist_state = persist_state

        self._tools_cache = build_tools_cache(tools)

        # Build extraction strategy
        extraction_strategy = create_extraction_strategy(
            state_mappers=state_mappers,
            compensation_schemas=compensation_schemas,
        )

        # Create tool executor
        executor = StrandsToolExecutor(self._tools_cache)

        # Create RecoveryManager
        self._rc_manager = RecoveryManager(
            compensation_pairs=compensation_pairs,
            alternative_map=alternative_map or {},
            retry_policy=retry_policy,
            extraction_strategy=extraction_strategy,
            action_executor=executor,
        )

        # State sync utility
        self._state_sync = StrandsStateSync()

        # Map tool_use_id -> record_id for tracking
        self._pending_records: dict[str, str] = {}

    @property
    def rc_manager(self) -> RecoveryManager:
        """Access to Core's RecoveryManager."""
        return self._rc_manager

    @property
    def transaction_log(self) -> TransactionLog:
        """Access to the transaction log."""
        return self._rc_manager.log

    def register_hooks(self, registry: Any) -> None:
        """Register hooks with Strands HookRegistry.

        This method is called by Strands when the agent is created.

        Args:
            registry: Strands HookRegistry instance
        """
        try:
            # Import Strands event types
            from strands.types.hooks import BeforeToolCallEvent, AfterToolCallEvent
        except ImportError:
            # Fallback for older versions or missing types
            logger.warning("Could not import Strands event types, using string registration")
            registry.add_callback("BeforeToolCallEvent", self._before_tool_call)
            registry.add_callback("AfterToolCallEvent", self._after_tool_call)
            return

        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)
        registry.add_callback(AfterToolCallEvent, self._after_tool_call)

    def _before_tool_call(self, event: Any) -> None:
        """Handle before tool call event.

        Records the action if the tool is compensatable.

        Event attributes:
        - selected_tool: The selected tool (modifiable)
        - tool_use: Dict with name, input, id
        - invocation_state: Dict for persisting state
        - cancel_tool: Set to True to cancel tool execution
        """
        tool_use = getattr(event, 'tool_use', {}) or {}
        tool_name = tool_use.get('name', '')
        tool_input = tool_use.get('input', {})
        tool_use_id = tool_use.get('id', '')

        if not tool_name:
            return

        # Load state from invocation_state if persisting
        if self._persist_state:
            invocation_state = getattr(event, 'invocation_state', {}) or {}
            self._state_sync.load_into_manager(invocation_state, self._rc_manager)

        # Record action if compensatable
        if self._rc_manager.is_compensatable(tool_name):
            record = self._rc_manager.record_action(tool_name, tool_input)
            self._pending_records[tool_use_id] = record.id
            logger.debug(f"[STRANDS] Recorded action: {tool_name} (id={record.id})")

    def _after_tool_call(self, event: Any) -> None:
        """Handle after tool call event.

        Checks for errors and handles recovery/rollback.

        Event attributes:
        - tool_use: Dict with name, input, id
        - result: ToolResult (modifiable)
        - exception: Exception if tool raised one
        - invocation_state: Dict for persisting state
        """
        tool_use = getattr(event, 'tool_use', {}) or {}
        tool_name = tool_use.get('name', '')
        tool_use_id = tool_use.get('id', '')
        result = getattr(event, 'result', None)
        exception = getattr(event, 'exception', None)

        if not tool_name:
            return

        # Get the record ID for this call
        record_id = self._pending_records.pop(tool_use_id, None)

        if not record_id:
            # Not a compensatable action
            return

        # Check for error
        is_error = False
        error_msg = ""

        if exception:
            is_error = True
            error_msg = str(exception)
        else:
            is_error, error_msg = self._detect_error(result)

        if is_error:
            logger.warning(f"[STRANDS] Tool {tool_name} failed: {error_msg}")
            self._rc_manager.mark_failed(record_id, error_msg)

            # Handle recovery and rollback
            new_result = self._handle_failure(tool_name, error_msg, record_id)

            # Modify event.result with compensation message
            self._modify_result(event, new_result)
        else:
            # Success - mark completed
            result_content = self._extract_result_content(result)
            self._rc_manager.mark_completed(record_id, result_content)
            logger.debug(f"[STRANDS] Marked completed: {tool_name}")

        # Save state to invocation_state if persisting
        if self._persist_state:
            invocation_state = getattr(event, 'invocation_state', {})
            if invocation_state is not None:
                self._state_sync.save_from_manager(invocation_state, self._rc_manager)

    def _detect_error(self, result: Any) -> tuple[bool, str]:
        """Detect if a result indicates an error."""
        if result is None:
            return False, ""

        # Check Strands result format
        if isinstance(result, dict):
            if result.get("status") == "error":
                content = result.get("content", [])
                if isinstance(content, list) and content:
                    return True, content[0].get("text", "Unknown error")
                return True, "Unknown error"

            # Check content for error
            content = result.get("content", [])
            if isinstance(content, list) and content:
                text = content[0].get("text", "")
                if text:
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict) and parsed.get("error"):
                            return True, str(parsed.get("error"))
                    except (json.JSONDecodeError, TypeError):
                        pass

        return False, ""

    def _extract_result_content(self, result: Any) -> Any:
        """Extract content from Strands result."""
        if result is None:
            return None

        if isinstance(result, dict):
            content = result.get("content", [])
            if isinstance(content, list) and content:
                text = content[0].get("text", "")
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    return text
            return content

        return result

    def _handle_failure(self, tool_name: str, error_msg: str, record_id: str) -> str:
        """Handle tool failure with recovery and rollback."""
        recovery_attempts = 0
        compensated_actions = []
        rollback_details = []
        failure_context_summary = ""

        # Try recovery
        if self._auto_recover:
            logger.info(f"[STRANDS] Attempting recovery for {tool_name}...")
            try:
                recovery_result = self._rc_manager.recover(record_id, error_msg)
                recovery_attempts = recovery_result.attempts

                if recovery_result.success:
                    logger.info(f"[STRANDS] Recovery succeeded via {recovery_result.action_taken}")
                    result = recovery_result.result
                    return str(result) if not isinstance(result, str) else result

            except Exception as recovery_error:
                logger.error(f"[STRANDS] Recovery error: {recovery_error}")

        # Get failure context
        try:
            failure_context_summary = self._rc_manager.get_failure_summary()
        except Exception:
            pass

        # Trigger rollback
        if self._auto_rollback:
            logger.info(f"[STRANDS] Triggering rollback for {tool_name}")
            try:
                rollback_plan = self._rc_manager.log.get_rollback_plan()
                for rec in rollback_plan:
                    rollback_details.append({
                        'action': rec.action,
                        'compensator': rec.compensator,
                        'params': rec.params,
                    })

                rollback_result = self._rc_manager.rollback()
                compensated_actions = getattr(rollback_result, 'compensated', [])
            except Exception as rollback_error:
                logger.error(f"[STRANDS] Rollback failed: {rollback_error}")

        # Return formatted message
        return format_compensation_message(
            failed_action=tool_name,
            error=error_msg,
            recovery_attempts=recovery_attempts,
            compensated_actions=compensated_actions,
            rollback_details=rollback_details,
            failure_context_summary=failure_context_summary,
            goals=self._goals,
        )

    def _modify_result(self, event: Any, message: str) -> None:
        """Modify event result with compensation message."""
        # Create new result in Strands format
        new_result = {
            "content": [{"type": "text", "text": message}],
            "status": "error",
        }

        # Try to get tool_use_id from event
        tool_use = getattr(event, 'tool_use', {}) or {}
        tool_use_id = tool_use.get('id')
        if tool_use_id:
            new_result["toolUseId"] = tool_use_id

        # Set the result on the event
        if hasattr(event, 'result'):
            event.result = new_result

    def rollback(self) -> Any:
        """Manually trigger rollback."""
        return self._rc_manager.rollback()

    def clear(self) -> None:
        """Clear the transaction log."""
        self._rc_manager.clear()
