"""Factory function for creating compensated LangChain agents.

Provides a convenient function to create LangChain/LangGraph agents
with compensation and recovery capabilities.

Recovery Flow:
1. Tool executes
2. On failure → Retry (configurable attempts with backoff)
3. If retries exhausted → Try alternatives (if configured)
4. If all recovery fails → Rollback all previous completed actions
5. Return informative message to LLM explaining what happened
6. LLM can retry from clean state with different approach
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, Callable, Sequence

from react_agent_compensation.core.config import AlternativeMap, CompensationPairs, RetryPolicy
from react_agent_compensation.core.extraction import CompensationSchema
from react_agent_compensation.core.models import ActionStatus
from react_agent_compensation.core.transaction_log import TransactionLog
from react_agent_compensation.langchain_adaptor.middleware import CompensationMiddleware

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


def _format_compensation_message(
    failed_action: str,
    error: str,
    recovery_attempts: int,
    compensated_actions: list[str],
    rollback_details: list[dict[str, Any]] | None = None,
) -> str:
    """Format an informative message for the LLM after compensation.

    This message tells the LLM what happened and that it can try again.

    Args:
        failed_action: Name of the tool that failed
        error: Error message
        recovery_attempts: Number of retry attempts made
        compensated_actions: List of record IDs that were compensated
        rollback_details: Optional details about what was rolled back

    Returns:
        Formatted message string for the LLM
    """
    lines = [
        f"[COMPENSATION TRIGGERED]",
        f"",
        f"Action '{failed_action}' failed after {recovery_attempts} retry attempt(s).",
        f"Error: {error}",
        f"",
    ]

    if compensated_actions:
        lines.append(f"Rolled back {len(compensated_actions)} previous action(s):")
        if rollback_details:
            for detail in rollback_details:
                action = detail.get('action', 'unknown')
                compensator = detail.get('compensator', 'unknown')
                params = detail.get('params', {})
                lines.append(f"  - {action}({params}) → compensated with {compensator}")
        else:
            for record_id in compensated_actions:
                lines.append(f"  - Record {record_id}")
        lines.append("")

    lines.extend([
        "State has been reset to before the failed sequence.",
        "You can now try a different approach or alternative parameters.",
    ])

    return "\n".join(lines)


def _wrap_tool_with_compensation(
    tool: Any,
    middleware: CompensationMiddleware,
    auto_rollback: bool = True,
    auto_recover: bool = True,
) -> Any:
    """Wrap a LangChain tool with compensation tracking and recovery.

    This creates a new tool that:
    1. Records the action before execution
    2. On failure: attempts recovery (retry + alternatives)
    3. If recovery fails: triggers rollback of all previous actions
    4. Returns informative message to LLM on failure (enabling retry)

    Args:
        tool: LangChain tool to wrap
        middleware: CompensationMiddleware instance for tracking
        auto_rollback: Whether to automatically rollback on unrecoverable failure
        auto_recover: Whether to automatically attempt recovery (retry/alternatives)

    Returns:
        Wrapped tool with same interface but compensation tracking
    """
    try:
        from langchain_core.tools import BaseTool, StructuredTool, tool as tool_decorator
    except ImportError as e:
        raise ImportError(
            "langchain-core is required. Install with: pip install langchain-core"
        ) from e

    tool_name = getattr(tool, 'name', str(tool))
    is_compensatable = middleware.rc_manager.is_compensatable(tool_name)

    # Get the original function
    if hasattr(tool, 'func'):
        original_func = tool.func
    elif hasattr(tool, '_run'):
        original_func = tool._run
    elif callable(tool):
        original_func = tool
    else:
        logger.warning(f"Cannot wrap tool {tool_name}: no callable found")
        return tool

    @functools.wraps(original_func)
    def wrapped_func(*args, **kwargs):
        """Wrapped tool function with compensation tracking and recovery."""
        # Convert args to kwargs using tool schema if available
        if args and hasattr(tool, 'args_schema'):
            schema = tool.args_schema
            if hasattr(schema, 'model_fields'):
                field_names = list(schema.model_fields.keys())
                for i, arg in enumerate(args):
                    if i < len(field_names):
                        kwargs[field_names[i]] = arg
                args = ()

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

            # Step 1: Try recovery (retry + alternatives)
            if auto_recover:
                logger.info(f"[COMPENSATION] Attempting recovery for {tool_name}...")
                try:
                    recovery_result = middleware.rc_manager.recover(record_id, error_msg)
                    recovery_attempts = recovery_result.attempts

                    if recovery_result.success:
                        # Recovery succeeded! Mark completed and return result
                        logger.info(
                            f"[COMPENSATION] Recovery succeeded for {tool_name} "
                            f"via {recovery_result.action_taken} after {recovery_attempts} attempt(s)"
                        )
                        return recovery_result.result

                    logger.warning(
                        f"[COMPENSATION] Recovery failed for {tool_name} "
                        f"after {recovery_attempts} attempt(s): {recovery_result.error}"
                    )
                except Exception as recovery_error:
                    logger.error(f"[COMPENSATION] Recovery error: {recovery_error}")

            # Step 2: Recovery failed - trigger rollback
            if auto_rollback:
                logger.info(f"[COMPENSATION] Triggering rollback due to unrecoverable failure in {tool_name}")
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
                    msg = getattr(rollback_result, 'message', 'completed') if rollback_result else 'completed'
                    logger.info(f"[COMPENSATION] Rollback completed: {msg}")

                except Exception as rollback_error:
                    logger.error(f"[COMPENSATION] Rollback failed: {rollback_error}")

            # Step 3: Return informative message to LLM
            return _format_compensation_message(
                failed_action=tool_name,
                error=error_msg,
                recovery_attempts=recovery_attempts,
                compensated_actions=compensated_actions,
                rollback_details=rollback_details,
            )

        try:
            # Execute the original tool
            if hasattr(tool, 'func') and tool.func:
                result = tool.func(*args, **kwargs)
            elif hasattr(tool, '_run'):
                result = tool._run(*args, **kwargs)
            else:
                result = original_func(*args, **kwargs)

            # Check if result indicates an error (some tools return error strings)
            is_error = False
            error_msg = ""
            if isinstance(result, str):
                error_indicators = ['error', 'failed', 'failure', 'exception', 'unavailable']
                is_error = any(ind in result.lower() for ind in error_indicators)
                if is_error:
                    error_msg = result
            elif isinstance(result, dict):
                is_error = result.get('error') or result.get('status') == 'failed'
                if is_error:
                    error_msg = result.get('error', str(result))

            if is_error and record:
                # Mark as failed first
                middleware.rc_manager.mark_failed(record.id, error_msg)
                logger.warning(f"[COMPENSATION] Tool {tool_name} returned error: {error_msg}")

                # Handle failure with recovery and rollback
                return _handle_failure(error_msg, record.id)

            # Success - mark completed
            if record:
                middleware.rc_manager.mark_completed(record.id, result)
                logger.debug(f"[COMPENSATION] Marked completed: {tool_name} (id={record.id})")

            return result

        except Exception as e:
            # Tool raised an exception
            error_msg = str(e)
            if record:
                middleware.rc_manager.mark_failed(record.id, error_msg)
                logger.error(f"[COMPENSATION] Tool {tool_name} failed with exception: {e}")

                # Handle failure with recovery and rollback
                # Return message instead of raising to let LLM continue
                return _handle_failure(error_msg, record.id)

            # If not compensatable, re-raise
            raise

    # Create new tool with wrapped function
    if isinstance(tool, StructuredTool):
        wrapped_tool = StructuredTool(
            name=tool.name,
            description=tool.description,
            func=wrapped_func,
            args_schema=tool.args_schema,
            return_direct=getattr(tool, 'return_direct', False),
        )
    elif isinstance(tool, BaseTool):
        # For other BaseTool subclasses, create a StructuredTool
        wrapped_tool = StructuredTool(
            name=tool.name,
            description=tool.description,
            func=wrapped_func,
            args_schema=getattr(tool, 'args_schema', None),
            return_direct=getattr(tool, 'return_direct', False),
        )
    else:
        # For function-based tools, use the tool decorator
        wrapped_tool = tool_decorator(wrapped_func)
        if hasattr(tool, 'name'):
            wrapped_tool.name = tool.name
        if hasattr(tool, 'description'):
            wrapped_tool.description = tool.description

    return wrapped_tool


def create_compensated_agent(
    model: str | Any,
    tools: Sequence[Any] | None = None,
    *,
    compensation_mapping: CompensationPairs,
    alternative_map: AlternativeMap | None = None,
    retry_policy: RetryPolicy | None = None,
    shared_log: TransactionLog | None = None,
    agent_id: str | None = None,
    compensation_schemas: dict[str, CompensationSchema] | None = None,
    state_mappers: dict[str, Callable] | None = None,
    use_llm_extraction: bool = False,
    llm_model: str = "gpt-4o-mini",
    checkpointer: Any = None,
    system_prompt: str | None = None,
    middleware: Sequence[Any] = (),
    response_format: Any = None,
    context_schema: Any = None,
    store: Any = None,
    debug: bool = False,
    name: str | None = None,
    cache: Any = None,
    auto_rollback: bool = True,
    auto_recover: bool = True,
) -> Any:
    """Create a LangChain agent with recovery and compensation.

    This function creates a LangGraph agent with automatic compensation
    tracking. Tools are wrapped to implement this recovery flow:

    1. Tool executes
    2. On failure → Retry (configurable via retry_policy)
    3. If retries exhausted → Try alternatives (from alternative_map)
    4. If all recovery fails → Rollback all previous completed actions
    5. Return informative message to LLM explaining what happened
    6. LLM can retry from clean state with different approach

    Args:
        model: LLM model (string or instance)
        tools: List of tools available to the agent
        compensation_mapping: Maps tool names to their compensation tools
        alternative_map: Maps tools to alternatives to try on failure
        retry_policy: Configuration for retry behavior (max_retries, backoff, etc.)
        shared_log: Shared TransactionLog for multi-agent scenarios
        agent_id: Identifier for this agent in multi-agent scenarios
        compensation_schemas: Declarative extraction schemas
        state_mappers: Custom extraction functions
        use_llm_extraction: Enable LLM-based parameter extraction
        llm_model: Model for LLM extraction
        checkpointer: LangGraph checkpointer for persistence
        system_prompt: System prompt for the agent
        middleware: Additional middleware (compensation added automatically)
        response_format: Response format configuration
        context_schema: Context schema for the agent
        store: Store for persistence
        debug: Enable debug mode
        name: Name for the agent
        cache: Cache configuration
        auto_rollback: Automatically rollback on unrecoverable failure (default: True)
        auto_recover: Automatically attempt recovery via retry/alternatives (default: True)

    Returns:
        Compiled LangGraph agent with compensation capabilities

    Example:
        from react_agent_compensation.core import RetryPolicy

        agent = create_compensated_agent(
            "gpt-4",
            tools=[book_flight, cancel_flight, book_flight_backup],
            compensation_mapping={"book_flight": "cancel_flight"},
            alternative_map={"book_flight": ["book_flight_backup"]},
            retry_policy=RetryPolicy(max_retries=3, base_delay=1.0),
        )
        result = agent.invoke({"messages": [("user", "Book a trip")]})

        # If book_flight fails:
        # 1. Retries up to 3 times with exponential backoff
        # 2. Tries book_flight_backup as alternative
        # 3. If all fail, calls cancel_flight for any completed bookings
        # 4. Returns message to LLM explaining what happened
        # 5. LLM can try a different approach
    """
    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError as e:
        raise ImportError(
            "LangGraph is required. Install with: pip install langgraph"
        ) from e

    # Create compensation middleware
    comp_middleware = CompensationMiddleware(
        compensation_mapping=compensation_mapping,
        tools=tools,
        alternative_map=alternative_map,
        retry_policy=retry_policy,
        shared_log=shared_log,
        agent_id=agent_id,
        compensation_schemas=compensation_schemas,
        state_mappers=state_mappers,
        use_llm_extraction=use_llm_extraction,
        llm_model=llm_model,
    )

    # Wrap tools with compensation tracking
    wrapped_tools = []
    if tools:
        for tool in tools:
            wrapped_tool = _wrap_tool_with_compensation(
                tool,
                comp_middleware,
                auto_rollback=auto_rollback,
                auto_recover=auto_recover,
            )
            wrapped_tools.append(wrapped_tool)
            tool_name = getattr(tool, 'name', str(tool))
            logger.debug(f"Wrapped tool: {tool_name}")

    logger.info(
        f"Creating compensated agent with {len(wrapped_tools)} tools, "
        f"{len(compensation_mapping)} compensation pairs, "
        f"retry_policy={retry_policy}, "
        f"alternatives={list(alternative_map.keys()) if alternative_map else []}"
    )

    # Create the agent with wrapped tools
    agent = create_react_agent(
        model,
        tools=wrapped_tools,
        checkpointer=checkpointer,
        prompt=system_prompt,
    )

    # Store middleware reference for access
    agent._compensation_middleware = comp_middleware

    logger.info(f"Created compensated agent with {len(compensation_mapping)} compensation pairs")

    return agent


def get_compensation_middleware(agent: Any) -> CompensationMiddleware | None:
    """Get the CompensationMiddleware from an agent.

    Args:
        agent: Agent created with create_compensated_agent

    Returns:
        CompensationMiddleware or None if not found
    """
    return getattr(agent, "_compensation_middleware", None)


def create_multi_agent_log() -> TransactionLog:
    """Create a shared TransactionLog for multi-agent scenarios.

    Use this when you have multiple agents that need coordinated
    rollback.

    Returns:
        New TransactionLog instance

    Example:
        shared_log = create_multi_agent_log()

        agent1 = create_compensated_agent(
            model, tools=tools1,
            compensation_mapping={...},
            shared_log=shared_log,
            agent_id="agent1",
        )

        agent2 = create_compensated_agent(
            model, tools=tools2,
            compensation_mapping={...},
            shared_log=shared_log,
            agent_id="agent2",
        )
    """
    return TransactionLog()
