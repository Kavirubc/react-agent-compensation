"""Factory functions for creating compensated AWS Strands agents.

Provides convenient functions to create Strands Agents with automatic
compensation and recovery capabilities.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

from react_agent_compensation.core.config import AlternativeMap, CompensationPairs, RetryPolicy
from react_agent_compensation.core.extraction import CompensationSchema
from react_agent_compensation.strands_adaptor.hooks import CompensationHookProvider

logger = logging.getLogger(__name__)


def create_compensated_agent(
    system_prompt: str | None = None,
    model: str | Any = None,
    tools: Sequence[Any] | None = None,
    *,
    compensation_mapping: CompensationPairs,
    alternative_map: AlternativeMap | None = None,
    retry_policy: RetryPolicy | None = None,
    compensation_schemas: dict[str, CompensationSchema] | None = None,
    state_mappers: dict[str, Callable] | None = None,
    goals: list[str] | None = None,
    additional_hooks: list[Any] | None = None,
    auto_rollback: bool = True,
    auto_recover: bool = True,
    **agent_kwargs: Any,
) -> Any:
    """Create a Strands Agent with compensation capabilities.

    Creates an agent with CompensationHookProvider that automatically
    tracks tool executions and provides recovery/rollback on failures.

    Args:
        system_prompt: System prompt for the agent
        model: Model to use (string name or model instance)
        tools: List of tools available to the agent
        compensation_mapping: Maps tool names to compensation tools
        alternative_map: Maps tools to alternatives to try on failure
        retry_policy: Configuration for retry behavior
        compensation_schemas: Declarative extraction schemas
        state_mappers: Custom extraction functions
        goals: Optimization goals for goal-aware recovery
        additional_hooks: Additional HookProviders to include
        auto_rollback: Automatically rollback on unrecoverable failure
        auto_recover: Automatically attempt recovery via retry/alternatives
        **agent_kwargs: Additional arguments passed to Agent constructor

    Returns:
        Strands Agent instance with compensation capabilities

    Example:
        agent = create_compensated_agent(
            system_prompt="You are an order processing assistant.",
            tools=[reserve_inventory, release_inventory, process_payment, refund_payment],
            compensation_mapping={
                "reserve_inventory": "release_inventory",
                "process_payment": "refund_payment",
            },
            retry_policy=RetryPolicy(max_retries=2),
        )

        result = agent("Process order for products SKU001, SKU002")
    """
    try:
        from strands import Agent
    except ImportError as e:
        raise ImportError(
            "strands-agents is required. Install with: pip install strands-agents"
        ) from e

    # Create compensation hook provider
    compensation_provider = CompensationHookProvider(
        compensation_pairs=compensation_mapping,
        tools=list(tools) if tools else None,
        alternative_map=alternative_map,
        retry_policy=retry_policy,
        compensation_schemas=compensation_schemas,
        state_mappers=state_mappers,
        goals=goals,
        auto_rollback=auto_rollback,
        auto_recover=auto_recover,
    )

    # Combine hooks
    hooks = [compensation_provider]
    if additional_hooks:
        hooks.extend(additional_hooks)

    # Build agent kwargs
    kwargs: dict[str, Any] = {}
    if system_prompt:
        kwargs["system_prompt"] = system_prompt
    if model:
        kwargs["model"] = model
    if tools:
        kwargs["tools"] = list(tools)
    kwargs["hooks"] = hooks
    kwargs.update(agent_kwargs)

    # Create agent
    agent = Agent(**kwargs)

    # Store provider reference for access if possible
    try:
        agent._compensation_provider = compensation_provider
    except AttributeError:
        # Some Agent implementations may use __slots__ or restrict dynamic attributes
        logger.debug("Could not attach provider reference to Agent instance")

    logger.info(
        f"Created compensated Strands agent with "
        f"{len(compensation_mapping)} compensation pairs"
    )

    return agent


async def create_compensated_agent_async(
    system_prompt: str | None = None,
    model: str | Any = None,
    tools: Sequence[Any] | None = None,
    *,
    compensation_mapping: CompensationPairs,
    alternative_map: AlternativeMap | None = None,
    retry_policy: RetryPolicy | None = None,
    compensation_schemas: dict[str, CompensationSchema] | None = None,
    state_mappers: dict[str, Callable] | None = None,
    goals: list[str] | None = None,
    additional_hooks: list[Any] | None = None,
    auto_rollback: bool = True,
    auto_recover: bool = True,
    **agent_kwargs: Any,
) -> Any:
    """Create a Strands Agent with compensation capabilities (async version).

    Same as create_compensated_agent but returns an async-ready agent.
    Use with agent.invoke_async() for async execution.

    See create_compensated_agent for full documentation.
    """
    # Same implementation - Strands agents support both sync and async
    return create_compensated_agent(
        system_prompt=system_prompt,
        model=model,
        tools=tools,
        compensation_mapping=compensation_mapping,
        alternative_map=alternative_map,
        retry_policy=retry_policy,
        compensation_schemas=compensation_schemas,
        state_mappers=state_mappers,
        goals=goals,
        additional_hooks=additional_hooks,
        auto_rollback=auto_rollback,
        auto_recover=auto_recover,
        **agent_kwargs,
    )


def wrap_tools_with_compensation(
    tools: Sequence[Any],
    compensation_mapping: CompensationPairs,
    *,
    alternative_map: AlternativeMap | None = None,
    retry_policy: RetryPolicy | None = None,
    compensation_schemas: dict[str, CompensationSchema] | None = None,
    state_mappers: dict[str, Callable] | None = None,
    goals: list[str] | None = None,
) -> tuple[CompensationHookProvider, list[Any]]:
    """Create a CompensationHookProvider for manual integration.

    Use this when you need more control over the agent setup or
    want to integrate with an existing agent.

    Args:
        tools: List of Strands tools
        compensation_mapping: Maps tool names to compensation tools
        alternative_map: Maps tools to alternatives to try on failure
        retry_policy: Configuration for retry behavior
        compensation_schemas: Declarative extraction schemas
        state_mappers: Custom extraction functions
        goals: Optimization goals for goal-aware recovery

    Returns:
        Tuple of (CompensationHookProvider, list of tools)

    Example:
        provider, tools = wrap_tools_with_compensation(
            tools=[reserve_inventory, release_inventory],
            compensation_mapping={"reserve_inventory": "release_inventory"},
        )

        agent = Agent(
            tools=tools,
            hooks=[provider],
        )
    """
    provider = CompensationHookProvider(
        compensation_pairs=compensation_mapping,
        tools=list(tools),
        alternative_map=alternative_map,
        retry_policy=retry_policy,
        compensation_schemas=compensation_schemas,
        state_mappers=state_mappers,
        goals=goals,
    )

    return provider, list(tools)


def get_compensation_provider(agent: Any) -> CompensationHookProvider | None:
    """Get the CompensationHookProvider from an agent.

    Args:
        agent: Agent created with create_compensated_agent

    Returns:
        CompensationHookProvider or None if not found
    """
    return getattr(agent, "_compensation_provider", None)
