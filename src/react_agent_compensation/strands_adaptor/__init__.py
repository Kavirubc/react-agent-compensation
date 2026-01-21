"""AWS Strands Agents SDK Adaptor for react-agent-compensation.

This module provides integration with AWS Strands Agents SDK,
wrapping the framework-agnostic Core module.

Quick Start:
    from react_agent_compensation.strands_adaptor import create_compensated_agent

    agent = create_compensated_agent(
        system_prompt="You are an order processing assistant.",
        tools=[reserve_inventory, release_inventory],
        compensation_mapping={"reserve_inventory": "release_inventory"},
    )
    result = agent("Process the order")

    # If a tool fails, all previous successful compensatable actions
    # are automatically rolled back using their compensation tools!

Manual Integration:
    from react_agent_compensation.strands_adaptor import (
        CompensationHookProvider,
        wrap_tools_with_compensation,
    )

    provider, tools = wrap_tools_with_compensation(
        tools=[reserve_inventory, release_inventory],
        compensation_mapping={"reserve_inventory": "release_inventory"},
    )

    agent = Agent(tools=tools, hooks=[provider])

Async Usage:
    agent = create_compensated_agent(...)
    result = await agent.invoke_async("Process the order")

Components:
- create_compensated_agent: Factory function for agents with automatic compensation
- CompensationHookProvider: HookProvider implementing Strands' hook protocol
- wrap_tools_with_compensation: Manual integration helper
- StrandsStateSync: State synchronization with invocation_state
- CompensationApprovalInterrupt: Optional human-in-the-loop approval
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
from react_agent_compensation.strands_adaptor.adapters import (
    SimpleActionResult,
    StrandsActionResult,
    StrandsToolExecutor,
    build_tools_cache,
)

# Factory functions
from react_agent_compensation.strands_adaptor.agent import (
    create_compensated_agent,
    create_compensated_agent_async,
    get_compensation_provider,
    wrap_tools_with_compensation,
)

# Hooks
from react_agent_compensation.strands_adaptor.hooks import (
    CompensationHookProvider,
    format_compensation_message,
)

# Interrupts
from react_agent_compensation.strands_adaptor.interrupts import (
    CompensationApprovalInterrupt,
    InteractiveApproval,
    create_logging_interrupt,
    create_webhook_interrupt,
)

# State management
from react_agent_compensation.strands_adaptor.state import (
    COMPENSATION_LOG_KEY,
    FAILURE_CONTEXT_KEY,
    StrandsStateSync,
    get_compensation_log,
    get_failure_context,
    sync_compensation_log,
    sync_failure_context,
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
    "StrandsActionResult",
    "StrandsToolExecutor",
    "SimpleActionResult",
    "build_tools_cache",
    # Hooks
    "CompensationHookProvider",
    "format_compensation_message",
    # Factory functions
    "create_compensated_agent",
    "create_compensated_agent_async",
    "wrap_tools_with_compensation",
    "get_compensation_provider",
    # Interrupts
    "CompensationApprovalInterrupt",
    "InteractiveApproval",
    "create_logging_interrupt",
    "create_webhook_interrupt",
    # State management
    "StrandsStateSync",
    "get_compensation_log",
    "sync_compensation_log",
    "get_failure_context",
    "sync_failure_context",
    "COMPENSATION_LOG_KEY",
    "FAILURE_CONTEXT_KEY",
]
