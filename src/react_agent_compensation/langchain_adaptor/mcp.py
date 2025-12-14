"""LangChain MCP integration with automatic compensation.

Provides a high-level factory function to create LangChain agents that
connect to MCP servers with automatic compensation/rollback support.

Example:
    agent, client = await create_compensated_mcp_agent(
        model=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
        mcp_servers={
            "family": {"url": "http://localhost:8000/sse", "transport": "sse"}
        },
        system_prompt="You are a family coordination assistant.",
    )

    # Run agent - compensation is automatic on failures
    result = await agent.ainvoke({
        "messages": [("user", "Add family member John as Dad")]
    })

    # Manual rollback if needed
    await client.rollback()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from react_agent_compensation.core.config import RetryPolicy
from react_agent_compensation.core.mcp import MCPCompensationClient

if TYPE_CHECKING:
    from react_agent_compensation.core.extraction.base import ExtractionStrategy

logger = logging.getLogger(__name__)


async def create_compensated_mcp_agent(
    model: str | Any,
    mcp_servers: dict[str, dict[str, Any]],
    *,
    system_prompt: str | None = None,
    retry_policy: RetryPolicy | None = None,
    extraction_strategy: "ExtractionStrategy | None" = None,
    additional_tools: list[Any] | None = None,
    checkpointer: Any = None,
) -> tuple[Any, MCPCompensationClient]:
    """Create a LangChain agent with MCP tools and automatic compensation.

    This function:
    1. Connects to configured MCP servers
    2. Discovers compensation pairs from tool annotations
    3. Wraps tools with compensation tracking
    4. Creates a LangGraph agent with the tools

    Args:
        model: LLM model (string or instance). Examples:
            - "gpt-4" or "openai:gpt-4" for OpenAI
            - ChatGoogleGenerativeAI instance for Gemini
            - Any LangChain chat model
        mcp_servers: MCP server configurations. Format:
            {
                "server_name": {
                    "url": "http://localhost:8000/sse",  # For SSE/HTTP
                    "transport": "sse",  # or "http", "stdio"
                    # For stdio:
                    "command": "python",
                    "args": ["server.py"],
                }
            }
        system_prompt: Optional system prompt for the agent
        retry_policy: Optional retry policy for recovery
        extraction_strategy: Optional strategy for extracting compensation params
        additional_tools: Additional LangChain tools to include
        checkpointer: Optional LangGraph checkpointer for persistence

    Returns:
        Tuple of (compiled_agent, mcp_client):
        - compiled_agent: LangGraph CompiledStateGraph ready to invoke
        - mcp_client: MCPCompensationClient for manual operations

    Example:
        from langchain_google_genai import ChatGoogleGenerativeAI

        agent, client = await create_compensated_mcp_agent(
            model=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
            mcp_servers={
                "family": {
                    "url": "http://localhost:8000/sse",
                    "transport": "sse",
                }
            },
            system_prompt="You are a helpful assistant.",
        )

        # Show compensation pairs
        pairs = await client.get_compensation_pairs()
        print(f"Discovered pairs: {pairs}")

        # Run agent
        result = await agent.ainvoke({
            "messages": [("user", "Add a family member named John")]
        })

        # Access transaction log
        log = client.recovery_manager.log.snapshot()

        # Manual rollback if needed
        await client.rollback()
    """
    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError as e:
        raise ImportError(
            "LangGraph is required. Install with: pip install langgraph"
        ) from e

    # Create and connect MCP client
    client = MCPCompensationClient(
        server_config=mcp_servers,
        retry_policy=retry_policy,
        extraction_strategy=extraction_strategy,
    )
    await client.connect()

    # Get wrapped tools
    mcp_tools = await client.get_tools()
    logger.info(f"Loaded {len(mcp_tools)} MCP tools with compensation tracking")

    # Combine with additional tools
    all_tools = list(mcp_tools)
    if additional_tools:
        all_tools.extend(additional_tools)

    # Create agent
    agent = create_react_agent(
        model,
        tools=all_tools,
        checkpointer=checkpointer,
        prompt=system_prompt,
    )

    # Log discovered compensation pairs
    pairs = await client.get_compensation_pairs()
    if pairs:
        logger.info(f"Agent created with {len(pairs)} compensation pairs:")
        for forward, compensator in pairs.items():
            logger.debug(f"  {forward} -> {compensator}")

    return agent, client


async def load_mcp_tools_with_compensation(
    mcp_servers: dict[str, dict[str, Any]],
    retry_policy: RetryPolicy | None = None,
) -> tuple[list[Any], MCPCompensationClient]:
    """Load MCP tools with compensation wrapping without creating an agent.

    Useful when you want to integrate MCP tools into an existing agent
    or custom workflow.

    Args:
        mcp_servers: MCP server configurations
        retry_policy: Optional retry policy

    Returns:
        Tuple of (tools, client)
    """
    client = MCPCompensationClient(
        server_config=mcp_servers,
        retry_policy=retry_policy,
    )
    await client.connect()

    tools = await client.get_tools()
    return tools, client
