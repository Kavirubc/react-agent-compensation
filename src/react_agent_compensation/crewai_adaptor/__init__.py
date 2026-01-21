"""CrewAI Adaptor for react-agent-compensation.

This module provides integration with CrewAI multi-agent orchestration
framework, wrapping the framework-agnostic Core module.

Quick Start:
    from react_agent_compensation.crewai_adaptor import create_compensated_crew

    crew = create_compensated_crew(
        agents=[travel_agent],
        tasks=[travel_task],
        compensation_mapping={"Book Flight": "Cancel Flight"},
    )
    result = crew.kickoff()

    # If a tool fails, all previous successful compensatable actions
    # are automatically rolled back using their compensation tools!

Manual Tool Wrapping:
    from react_agent_compensation.crewai_adaptor import (
        wrap_tool_with_compensation,
        CrewAICompensationMiddleware,
    )

    middleware = CrewAICompensationMiddleware(
        compensation_mapping={"Book Flight": "Cancel Flight"},
        tools=[book_flight, cancel_flight],
    )
    wrapped_tool = wrap_tool_with_compensation(book_flight, middleware)

Hook-Based Integration:
    from react_agent_compensation.crewai_adaptor import (
        CrewAIHookManager,
        create_compensation_hooks,
    )

    before_hook, after_hook = create_compensation_hooks(middleware)
    # Register with CrewAI's @before_tool_call / @after_tool_call

Components:
- create_compensated_crew: Factory function for crews with automatic compensation
- create_compensated_agent: Factory function for agents with compensation-wrapped tools
- wrap_tool_with_compensation: Wrap individual tools with compensation tracking
- CrewAICompensationMiddleware: Middleware for recovery/compensation
- CrewAIHookManager: Hook-based integration manager
- CrewAIStateSync: State synchronization utilities
"""

# Re-export Core components for convenience
from react_agent_compensation.core import (
    ActionRecord,
    ActionStatus,
    AlternativeMap,
    CompensationPairs,
    RecoveryManager,
    RetryPolicy,
    RollbackFailure,
    TransactionLog,
)
from react_agent_compensation.core.extraction import CompensationSchema

# Adapters
from react_agent_compensation.crewai_adaptor.adapters import (
    CrewAIActionResult,
    CrewAIToolExecutor,
    SimpleActionResult,
    build_tools_cache,
)

# Factory functions
from react_agent_compensation.crewai_adaptor.factory import (
    create_compensated_agent,
    create_compensated_crew,
    create_shared_log,
    get_compensation_middleware,
)

# Hooks
from react_agent_compensation.crewai_adaptor.hooks import (
    CrewAIHookManager,
    create_compensation_hooks,
)

# Middleware
from react_agent_compensation.crewai_adaptor.middleware import CrewAICompensationMiddleware

# State management
from react_agent_compensation.crewai_adaptor.state import (
    COMPENSATION_LOG_KEY,
    FAILURE_CONTEXT_KEY,
    CrewAIStateSync,
    create_shared_state,
    get_compensation_log,
    get_failure_context,
    sync_compensation_log,
    sync_failure_context,
)

# Tool wrapping
from react_agent_compensation.crewai_adaptor.tool_wrapper import (
    format_compensation_message,
    wrap_tool_with_compensation,
)

__version__ = "0.1.0"

__all__ = [
    # Core re-exports
    "ActionRecord",
    "ActionStatus",
    "TransactionLog",
    "RecoveryManager",
    "RetryPolicy",
    "CompensationPairs",
    "AlternativeMap",
    "RollbackFailure",
    "CompensationSchema",
    # Adapters
    "CrewAIActionResult",
    "CrewAIToolExecutor",
    "SimpleActionResult",
    "build_tools_cache",
    # Middleware
    "CrewAICompensationMiddleware",
    # Factory functions
    "create_compensated_crew",
    "create_compensated_agent",
    "create_shared_log",
    "get_compensation_middleware",
    # Tool wrapping
    "wrap_tool_with_compensation",
    "format_compensation_message",
    # Hooks
    "CrewAIHookManager",
    "create_compensation_hooks",
    # State management
    "CrewAIStateSync",
    "get_compensation_log",
    "sync_compensation_log",
    "get_failure_context",
    "sync_failure_context",
    "create_shared_state",
    "COMPENSATION_LOG_KEY",
    "FAILURE_CONTEXT_KEY",
]
