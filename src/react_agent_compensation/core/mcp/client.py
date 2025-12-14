"""MCP Compensation Client.

High-level client that wraps langchain-mcp-adapters to provide
automatic compensation discovery and tool wrapping.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from react_agent_compensation.core.config import CompensationPairs, RetryPolicy
from react_agent_compensation.core.mcp.parser import discover_compensation_pairs
from react_agent_compensation.core.mcp.tools import CompensatedMCPTool, wrap_mcp_tools
from react_agent_compensation.core.recovery_manager import RecoveryManager

if TYPE_CHECKING:
    from react_agent_compensation.core.extraction.base import ExtractionStrategy
    from react_agent_compensation.core.protocols import ActionExecutor

logger = logging.getLogger(__name__)


class MCPCompensationClient:
    """High-level MCP client with automatic compensation support.

    Wraps langchain-mcp-adapters to:
    1. Connect to MCP servers
    2. Discover compensation pairs from tool annotations
    3. Return compensated tools ready for LangChain agents

    Example:
        client = MCPCompensationClient({
            "family": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        })
        await client.connect()

        tools = await client.get_tools()
        # tools are wrapped with compensation tracking

        pairs = await client.get_compensation_pairs()
        # {"add_family_member": "delete_family_member", ...}
    """

    def __init__(
        self,
        server_config: dict[str, Any],
        retry_policy: RetryPolicy | None = None,
        extraction_strategy: "ExtractionStrategy | None" = None,
        action_executor: "ActionExecutor | None" = None,
    ):
        """Initialize MCP compensation client.

        Args:
            server_config: Configuration for MCP servers (MultiServerMCPClient format)
            retry_policy: Optional retry policy for recovery
            extraction_strategy: Optional extraction strategy for compensation params
            action_executor: Optional executor for compensation actions
        """
        self._server_config = server_config
        self._retry_policy = retry_policy
        self._extraction_strategy = extraction_strategy
        self._action_executor = action_executor

        self._mcp_client: Any = None
        self._raw_tools: list[Any] = []
        self._tool_schemas: dict[str, dict[str, Any]] = {}
        self._compensation_pairs: CompensationPairs = {}
        self._recovery_manager: RecoveryManager | None = None
        self._wrapped_tools: list[CompensatedMCPTool] = []
        self._connected = False

    @property
    def recovery_manager(self) -> RecoveryManager:
        """Get the recovery manager for manual operations."""
        if not self._recovery_manager:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._recovery_manager

    @property
    def connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    async def connect(self) -> None:
        """Connect to MCP servers and discover compensation pairs.

        This:
        1. Connects to all configured MCP servers
        2. Fetches tool schemas and extracts compensation pairs
        3. Creates RecoveryManager with discovered pairs
        4. Wraps tools with compensation tracking
        """
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as e:
            raise ImportError(
                "langchain-mcp-adapters is required for MCP integration. "
                "Install with: pip install langchain-mcp-adapters"
            ) from e

        # Create MCP client
        self._mcp_client = MultiServerMCPClient(self._server_config)

        # Get tools
        self._raw_tools = await self._mcp_client.get_tools()
        logger.info(f"Loaded {len(self._raw_tools)} tools from MCP servers")

        # Discover compensation pairs from tool schemas
        await self._discover_pairs()

        # Create recovery manager
        self._recovery_manager = RecoveryManager(
            compensation_pairs=self._compensation_pairs,
            retry_policy=self._retry_policy,
            extraction_strategy=self._extraction_strategy,
            action_executor=self._action_executor or self._create_executor(),
        )

        # Wrap tools
        self._wrapped_tools = wrap_mcp_tools(
            self._raw_tools,
            self._recovery_manager,
            self._tool_schemas,
        )

        self._connected = True
        logger.info(
            f"MCP client connected with {len(self._compensation_pairs)} compensation pairs"
        )

    async def _discover_pairs(self) -> None:
        """Discover compensation pairs from MCP tool schemas."""
        # Try to get raw schemas from MCP sessions for full annotation access
        schemas = await self._fetch_raw_schemas()

        if schemas:
            # Use raw MCP schemas (have full annotations)
            self._compensation_pairs = discover_compensation_pairs(schemas)
            self._tool_schemas = {s["name"]: s.get("annotations", {}) for s in schemas}
        else:
            # Fallback to LangChain tool objects
            self._compensation_pairs = discover_compensation_pairs(self._raw_tools)

        logger.debug(f"Discovered compensation pairs: {self._compensation_pairs}")

    async def _fetch_raw_schemas(self) -> list[dict[str, Any]]:
        """Fetch raw MCP tool schemas from server sessions.

        This gets the full schema including annotations, which may not
        be available on the LangChain tool wrappers.
        """
        schemas = []

        try:
            # Access internal sessions from MultiServerMCPClient
            if hasattr(self._mcp_client, "_sessions"):
                for server_name, session in self._mcp_client._sessions.items():
                    try:
                        # List tools from MCP session
                        tools_result = await session.list_tools()
                        for tool in tools_result.tools:
                            schema = {
                                "name": tool.name,
                                "description": tool.description or "",
                                "inputSchema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
                                "annotations": getattr(tool, "annotations", {}) or {},
                            }
                            schemas.append(schema)
                    except Exception as e:
                        logger.debug(f"Could not fetch schemas from {server_name}: {e}")
        except Exception as e:
            logger.debug(f"Could not access MCP sessions: {e}")

        return schemas

    def _create_executor(self) -> "ActionExecutor":
        """Create an executor that invokes tools via MCP."""
        return MCPToolExecutor(self)

    async def get_tools(self) -> list[CompensatedMCPTool]:
        """Get all tools wrapped with compensation support.

        Returns:
            List of CompensatedMCPTool instances

        Raises:
            RuntimeError: If not connected
        """
        if not self._connected:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._wrapped_tools

    async def get_compensation_pairs(self) -> CompensationPairs:
        """Get discovered compensation pairs.

        Returns:
            Dict mapping tool names to compensation tool names
        """
        return self._compensation_pairs.copy()

    async def get_tool_metadata(self, tool_name: str) -> dict[str, Any]:
        """Get metadata for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool metadata (x-action-type, x-reversible, etc.)
        """
        return self._tool_schemas.get(tool_name, {})

    async def rollback(self) -> Any:
        """Execute rollback for all completed compensatable actions.

        Returns:
            RollbackResult with details
        """
        if not self._recovery_manager:
            raise RuntimeError("Client not connected")
        return self._recovery_manager.rollback()

    async def close(self) -> None:
        """Close the MCP client connection."""
        if self._mcp_client and hasattr(self._mcp_client, "close"):
            await self._mcp_client.close()
        self._connected = False


class MCPToolExecutor:
    """ActionExecutor that invokes tools via MCP.

    Used by RecoveryManager to execute compensation actions
    through the MCP protocol.
    """

    def __init__(self, client: MCPCompensationClient):
        self._client = client

    def execute(self, action: str, params: dict[str, Any]) -> Any:
        """Execute a tool action.

        Note: This is synchronous but wraps async internally.
        For async usage, use execute_async.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, schedule coroutine
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.execute_async(action, params))
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(self.execute_async(action, params))

    async def execute_async(self, action: str, params: dict[str, Any]) -> Any:
        """Execute a tool action asynchronously."""
        # Find the tool
        for tool in self._client._raw_tools:
            if tool.name == action:
                return await tool.ainvoke(params)

        raise ValueError(f"Tool not found: {action}")
