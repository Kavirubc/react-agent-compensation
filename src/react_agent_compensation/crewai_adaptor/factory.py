"""Factory functions for creating compensated CrewAI agents and crews.

Provides convenient functions to create CrewAI Agents and Crews
with automatic compensation and recovery capabilities.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Sequence

from react_agent_compensation.core.config import AlternativeMap, CompensationPairs, RetryPolicy
from react_agent_compensation.core.extraction import CompensationSchema
from react_agent_compensation.core.transaction_log import TransactionLog
from react_agent_compensation.crewai_adaptor.middleware import CrewAICompensationMiddleware
from react_agent_compensation.crewai_adaptor.tool_wrapper import wrap_tool_with_compensation

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


def create_compensated_crew(
    agents: list[Any],
    tasks: list[Any],
    compensation_mapping: CompensationPairs,
    *,
    alternative_map: AlternativeMap | None = None,
    retry_policy: RetryPolicy | None = None,
    shared_log: TransactionLog | None = None,
    compensation_schemas: dict[str, CompensationSchema] | None = None,
    state_mappers: dict[str, Callable] | None = None,
    goals: list[str] | None = None,
    verbose: bool = False,
    **crew_kwargs: Any,
) -> Any:
    """Create a CrewAI Crew with compensation-wrapped tools on all agents.

    All agent tools are automatically wrapped with compensation tracking.
    When a tool fails, automatic recovery is attempted. If recovery fails,
    all previous completed compensatable actions are rolled back.

    Args:
        agents: List of CrewAI Agent instances
        tasks: List of CrewAI Task instances
        compensation_mapping: Maps tool names to compensation tools
        alternative_map: Maps tools to alternatives to try on failure
        retry_policy: Configuration for retry behavior
        shared_log: Shared TransactionLog for multi-agent coordination
        compensation_schemas: Declarative extraction schemas
        state_mappers: Custom extraction functions
        goals: Optimization goals for goal-aware recovery
        verbose: Enable verbose mode for the crew
        **crew_kwargs: Additional arguments passed to Crew constructor

    Returns:
        CrewAI Crew instance with compensation capabilities

    Example:
        crew = create_compensated_crew(
            agents=[travel_agent],
            tasks=[travel_task],
            compensation_mapping={
                "Book Flight": "Cancel Flight",
                "Book Hotel": "Cancel Hotel",
            },
            goals=["minimize_cost", "prefer_direct_flights"],
            verbose=True,
        )
        result = crew.kickoff()
    """
    try:
        from crewai import Crew
    except ImportError as e:
        raise ImportError(
            "crewai is required. Install with: pip install crewai"
        ) from e

    # Create shared log if not provided for multi-agent coordination
    if shared_log is None:
        shared_log = TransactionLog()

    # Collect all tools from all agents
    all_tools = []
    for agent in agents:
        if hasattr(agent, "tools") and agent.tools:
            all_tools.extend(agent.tools)

    # Create middleware with all tools
    middleware = CrewAICompensationMiddleware(
        compensation_mapping=compensation_mapping,
        tools=all_tools,
        alternative_map=alternative_map,
        retry_policy=retry_policy,
        shared_log=shared_log,
        compensation_schemas=compensation_schemas,
        state_mappers=state_mappers,
        goals=goals,
    )

    # Wrap each agent's tools
    for agent in agents:
        if hasattr(agent, "tools") and agent.tools:
            wrapped_tools = []
            for tool in agent.tools:
                wrapped_tool = wrap_tool_with_compensation(
                    tool,
                    middleware,
                    auto_rollback=True,
                    auto_recover=True,
                )
                wrapped_tools.append(wrapped_tool)
            agent.tools = wrapped_tools

    # Create the crew
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=verbose,
        **crew_kwargs,
    )

    # Store middleware reference for access
    crew._compensation_middleware = middleware

    logger.info(
        f"Created compensated crew with {len(agents)} agents, "
        f"{len(compensation_mapping)} compensation pairs"
    )

    return crew


def create_compensated_agent(
    role: str,
    goal: str,
    backstory: str,
    tools: Sequence[Any],
    compensation_mapping: CompensationPairs,
    *,
    alternative_map: AlternativeMap | None = None,
    retry_policy: RetryPolicy | None = None,
    shared_log: TransactionLog | None = None,
    agent_id: str | None = None,
    compensation_schemas: dict[str, CompensationSchema] | None = None,
    state_mappers: dict[str, Callable] | None = None,
    goals: list[str] | None = None,
    verbose: bool = False,
    **agent_kwargs: Any,
) -> Any:
    """Create a CrewAI Agent with compensation-wrapped tools.

    Args:
        role: Agent's role
        goal: Agent's goal
        backstory: Agent's backstory
        tools: List of tools for the agent
        compensation_mapping: Maps tool names to compensation tools
        alternative_map: Maps tools to alternatives to try on failure
        retry_policy: Configuration for retry behavior
        shared_log: Shared TransactionLog for multi-agent scenarios
        agent_id: Unique identifier for this agent
        compensation_schemas: Declarative extraction schemas
        state_mappers: Custom extraction functions
        goals: Optimization goals for goal-aware recovery
        verbose: Enable verbose mode
        **agent_kwargs: Additional arguments passed to Agent constructor

    Returns:
        CrewAI Agent instance with compensation capabilities

    Example:
        agent = create_compensated_agent(
            role="Travel Agent",
            goal="Book complete travel arrangements",
            backstory="Expert travel agent...",
            tools=[book_flight, cancel_flight, book_hotel, cancel_hotel],
            compensation_mapping={
                "Book Flight": "Cancel Flight",
                "Book Hotel": "Cancel Hotel",
            },
        )
    """
    try:
        from crewai import Agent
    except ImportError as e:
        raise ImportError(
            "crewai is required. Install with: pip install crewai"
        ) from e

    # Create middleware
    middleware = CrewAICompensationMiddleware(
        compensation_mapping=compensation_mapping,
        tools=list(tools),
        alternative_map=alternative_map,
        retry_policy=retry_policy,
        shared_log=shared_log,
        agent_id=agent_id,
        compensation_schemas=compensation_schemas,
        state_mappers=state_mappers,
        goals=goals,
    )

    # Wrap tools
    wrapped_tools = []
    for tool in tools:
        wrapped_tool = wrap_tool_with_compensation(
            tool,
            middleware,
            auto_rollback=True,
            auto_recover=True,
        )
        wrapped_tools.append(wrapped_tool)

    # Create agent
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=wrapped_tools,
        verbose=verbose,
        **agent_kwargs,
    )

    # Store middleware reference
    agent._compensation_middleware = middleware

    logger.info(
        f"Created compensated agent '{role}' with "
        f"{len(compensation_mapping)} compensation pairs"
    )

    return agent


def get_compensation_middleware(agent_or_crew: Any) -> CrewAICompensationMiddleware | None:
    """Get the CompensationMiddleware from an agent or crew.

    Args:
        agent_or_crew: Agent or Crew created with compensation functions

    Returns:
        CrewAICompensationMiddleware or None if not found
    """
    return getattr(agent_or_crew, "_compensation_middleware", None)


def create_shared_log() -> TransactionLog:
    """Create a shared TransactionLog for multi-agent/crew scenarios.

    Returns:
        New TransactionLog instance

    Example:
        shared_log = create_shared_log()

        agent1 = create_compensated_agent(
            ...,
            shared_log=shared_log,
            agent_id="agent1",
        )

        agent2 = create_compensated_agent(
            ...,
            shared_log=shared_log,
            agent_id="agent2",
        )
    """
    return TransactionLog()
