"""Factory function for creating compensated LangChain agents.

Provides a convenient function to create LangChain/LangGraph agents
with compensation and recovery capabilities.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Sequence

from react_agent_compensation.core.config import AlternativeMap, CompensationPairs, RetryPolicy
from react_agent_compensation.core.extraction import CompensationSchema
from react_agent_compensation.core.transaction_log import TransactionLog
from react_agent_compensation.langchain_adaptor.middleware import CompensationMiddleware

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


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
) -> Any:
    """Create a LangChain agent with recovery and compensation.

    This function creates a LangGraph agent with CompensationMiddleware
    automatically configured for recovery and rollback.

    Args:
        model: LLM model (string or instance)
        tools: List of tools available to the agent
        compensation_mapping: Maps tool names to their compensation tools
        alternative_map: Maps tools to alternatives to try on failure
        retry_policy: Configuration for retry behavior
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

    Returns:
        Compiled LangGraph agent with compensation capabilities

    Example:
        agent = create_compensated_agent(
            "gpt-4",
            tools=[book_flight, cancel_flight],
            compensation_mapping={"book_flight": "cancel_flight"},
            alternative_map={"book_flight": ["book_flight_backup"]},
        )
        result = agent.invoke({"messages": [("user", "Book a trip")]})
    """
    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError as e:
        raise ImportError(
            "LangGraph is required. Install with: pip install langgraph"
        ) from e

    # Build middleware stack
    agent_middleware = list(middleware)

    # Add compensation middleware
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
    agent_middleware.append(comp_middleware)

    # Create the agent
    agent = create_react_agent(
        model,
        tools=tools,
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
