"""LangChain Adaptor for react-agent-compensation.

This module provides integration with LangChain/LangGraph agents,
wrapping the framework-agnostic Core module.

Quick Start:
    from react_agent_compensation.langchain_adaptor import create_compensated_agent

    agent = create_compensated_agent(
        model="gpt-4",
        tools=[book_flight, cancel_flight],
        compensation_mapping={"book_flight": "cancel_flight"},
    )
    result = agent.invoke({"messages": [("user", "Book me a flight")]})

MCP Integration:
    from react_agent_compensation.langchain_adaptor import create_compensated_mcp_agent

    agent, client = await create_compensated_mcp_agent(
        model="gpt-4",
        mcp_servers={"server": {"url": "http://localhost:8000/sse", "transport": "sse"}},
    )
    result = await agent.ainvoke({"messages": [("user", "Add item")]})

Components:
- CompensationMiddleware: LangChain middleware for recovery/compensation
- ToolCallInterceptor: Intercepts tool calls with recovery handling
- LangGraphStateSync: Synchronizes transaction log with LangGraph state
- create_compensated_agent: Factory function for agents with compensation
- create_compensated_mcp_agent: Factory for MCP-connected agents with compensation
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

# Adapters
from react_agent_compensation.langchain_adaptor.adapters import (
    LangChainActionResult,
    LangChainToolExecutor,
    LangChainToolSchema,
    SimpleActionResult,
    build_tool_schemas,
    build_tools_cache,
)

# Agent factory
from react_agent_compensation.langchain_adaptor.agent import (
    create_compensated_agent,
    create_multi_agent_log,
    get_compensation_middleware,
)

# Interceptors
from react_agent_compensation.langchain_adaptor.interceptors import (
    InterceptResult,
    ToolCallInterceptor,
)

# Middleware
from react_agent_compensation.langchain_adaptor.middleware import CompensationMiddleware

# State management
from react_agent_compensation.langchain_adaptor.state import (
    ACTION_LOG_KEY,
    LangGraphStateSync,
    create_shared_log,
    get_action_log,
    sync_action_log,
)

# MCP integration
from react_agent_compensation.langchain_adaptor.mcp import (
    create_compensated_mcp_agent,
    load_mcp_tools_with_compensation,
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
    # Adapters
    "LangChainActionResult",
    "LangChainToolExecutor",
    "LangChainToolSchema",
    "SimpleActionResult",
    "build_tools_cache",
    "build_tool_schemas",
    # Interceptors
    "ToolCallInterceptor",
    "InterceptResult",
    # Middleware
    "CompensationMiddleware",
    # Agent factory
    "create_compensated_agent",
    "create_multi_agent_log",
    "get_compensation_middleware",
    # MCP integration
    "create_compensated_mcp_agent",
    "load_mcp_tools_with_compensation",
    # State management
    "LangGraphStateSync",
    "get_action_log",
    "sync_action_log",
    "create_shared_log",
    "ACTION_LOG_KEY",
]
