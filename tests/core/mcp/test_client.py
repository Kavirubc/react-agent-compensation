"""Tests for MCPCompensationClient."""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch, Mock

import pytest

from react_agent_compensation.core.config import RetryPolicy
from react_agent_compensation.core.models import ActionStatus
from react_agent_compensation.core.mcp.client import MCPCompensationClient, MCPToolExecutor
from react_agent_compensation.core.mcp.tools import CompensatedMCPTool

# Check if langchain_mcp_adapters is available
try:
    import langchain_mcp_adapters
    HAS_MCP_ADAPTERS = True
except ImportError:
    HAS_MCP_ADAPTERS = False

requires_mcp_adapters = pytest.mark.skipif(
    not HAS_MCP_ADAPTERS,
    reason="langchain-mcp-adapters not installed"
)


class MockMCPTool:
    """Mock MCP tool for testing."""

    def __init__(self, name: str, description: str = "Test tool"):
        self.name = name
        self.description = description

    def invoke(self, input: dict, config: dict | None = None) -> str:
        return json.dumps({"status": "ok"})

    async def ainvoke(self, input: dict, config: dict | None = None) -> str:
        return json.dumps({"status": "ok"})


class MockToolSchema:
    """Mock MCP tool schema with annotations."""

    def __init__(self, name: str, annotations: dict | None = None):
        self.name = name
        self.description = f"Description for {name}"
        self.inputSchema = {}
        self.annotations = annotations or {}


class MockToolsResult:
    """Mock result from list_tools()."""

    def __init__(self, tools: list):
        self.tools = tools


class MockSession:
    """Mock MCP session."""

    def __init__(self, tools: list):
        self._tools = tools

    async def list_tools(self):
        return MockToolsResult(self._tools)


class MockMultiServerMCPClient:
    """Mock MultiServerMCPClient."""

    def __init__(self, server_config: dict):
        self._server_config = server_config
        self._sessions = {}
        self._tools = []

    async def get_tools(self) -> list:
        return self._tools

    async def close(self):
        pass


class TestMCPCompensationClientBasics:
    """Basic tests for MCPCompensationClient."""

    def test_create_client(self):
        """Test creating a client."""
        client = MCPCompensationClient(
            server_config={
                "test": {"url": "http://localhost:8000/sse", "transport": "sse"}
            }
        )

        assert client.connected is False

    def test_not_connected_raises(self):
        """Test accessing recovery_manager before connect raises."""
        client = MCPCompensationClient(server_config={})

        with pytest.raises(RuntimeError, match="not connected"):
            _ = client.recovery_manager

    @pytest.mark.asyncio
    async def test_get_tools_before_connect_raises(self):
        """Test get_tools before connect raises."""
        client = MCPCompensationClient(server_config={})

        with pytest.raises(RuntimeError, match="not connected"):
            await client.get_tools()


@requires_mcp_adapters
class TestMCPCompensationClientConnect:
    """Tests for client connection and tool discovery."""

    @pytest.mark.asyncio
    async def test_connect_discovers_tools(self):
        """Test that connect discovers tools from MCP server."""
        mock_tools = [
            MockMCPTool("add_item"),
            MockMCPTool("delete_item"),
        ]

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient"
        ) as MockClient:
            mock_client = MockMultiServerMCPClient({})
            mock_client._tools = mock_tools
            MockClient.return_value = mock_client

            client = MCPCompensationClient(
                server_config={"test": {"url": "http://localhost:8000/sse"}}
            )
            await client.connect()

            assert client.connected is True
            tools = await client.get_tools()
            assert len(tools) == 2
            assert all(isinstance(t, CompensatedMCPTool) for t in tools)

    @pytest.mark.asyncio
    async def test_connect_discovers_compensation_pairs(self):
        """Test that connect discovers compensation pairs from annotations."""
        mock_tools = [
            MockMCPTool("add_item"),
            MockMCPTool("delete_item"),
        ]
        mock_schemas = [
            MockToolSchema("add_item", {"x-compensation-pair": "delete_item"}),
            MockToolSchema("delete_item", {"x-compensation-pair": "add_item"}),
        ]

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient"
        ) as MockClient:
            mock_client = MockMultiServerMCPClient({})
            mock_client._tools = mock_tools
            mock_client._sessions = {
                "test": MockSession(mock_schemas),
            }
            MockClient.return_value = mock_client

            client = MCPCompensationClient(
                server_config={"test": {"url": "http://localhost:8000/sse"}}
            )
            await client.connect()

            pairs = await client.get_compensation_pairs()
            assert pairs == {"add_item": "delete_item", "delete_item": "add_item"}

    @pytest.mark.asyncio
    async def test_connect_with_retry_policy(self):
        """Test connecting with custom retry policy."""
        mock_tools = [MockMCPTool("test_tool")]

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient"
        ) as MockClient:
            mock_client = MockMultiServerMCPClient({})
            mock_client._tools = mock_tools
            MockClient.return_value = mock_client

            policy = RetryPolicy(max_retries=5, initial_delay=0.5)
            client = MCPCompensationClient(
                server_config={"test": {"url": "http://localhost:8000/sse"}},
                retry_policy=policy,
            )
            await client.connect()

            assert client.recovery_manager.retry_policy.max_retries == 5


@requires_mcp_adapters
class TestMCPCompensationClientOperations:
    """Tests for client operations after connect."""

    @pytest.mark.asyncio
    async def test_get_tool_metadata(self):
        """Test getting metadata for a specific tool."""
        mock_tools = [MockMCPTool("add_item")]
        mock_schemas = [
            MockToolSchema("add_item", {
                "x-compensation-pair": "delete_item",
                "x-action-type": "create",
            }),
        ]

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient"
        ) as MockClient:
            mock_client = MockMultiServerMCPClient({})
            mock_client._tools = mock_tools
            mock_client._sessions = {"test": MockSession(mock_schemas)}
            MockClient.return_value = mock_client

            client = MCPCompensationClient(
                server_config={"test": {"url": "http://localhost:8000/sse"}}
            )
            await client.connect()

            metadata = await client.get_tool_metadata("add_item")
            assert metadata.get("x-action-type") == "create"

    @pytest.mark.asyncio
    async def test_rollback_delegates_to_manager(self):
        """Test that rollback delegates to recovery manager."""
        mock_tools = [MockMCPTool("add_item")]

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient"
        ) as MockClient:
            mock_client = MockMultiServerMCPClient({})
            mock_client._tools = mock_tools
            MockClient.return_value = mock_client

            client = MCPCompensationClient(
                server_config={"test": {"url": "http://localhost:8000/sse"}}
            )
            await client.connect()

            # No actions to rollback
            result = await client.rollback()
            assert result.success is True

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the client."""
        mock_tools = [MockMCPTool("test_tool")]

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient"
        ) as MockClient:
            mock_client = MockMultiServerMCPClient({})
            mock_client._tools = mock_tools
            MockClient.return_value = mock_client

            client = MCPCompensationClient(
                server_config={"test": {"url": "http://localhost:8000/sse"}}
            )
            await client.connect()
            assert client.connected is True

            await client.close()
            assert client.connected is False


@requires_mcp_adapters
class TestMCPCompensationClientIntegration:
    """Integration tests for tool tracking."""

    @pytest.mark.asyncio
    async def test_tool_invocation_tracked(self):
        """Test that tool invocations are tracked in transaction log."""
        mock_tools = [
            MockMCPTool("add_item"),
        ]
        mock_schemas = [
            MockToolSchema("add_item", {"x-compensation-pair": "delete_item"}),
        ]

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient"
        ) as MockClient:
            mock_client = MockMultiServerMCPClient({})
            mock_client._tools = mock_tools
            mock_client._sessions = {"test": MockSession(mock_schemas)}
            MockClient.return_value = mock_client

            client = MCPCompensationClient(
                server_config={"test": {"url": "http://localhost:8000/sse"}}
            )
            await client.connect()

            tools = await client.get_tools()
            add_tool = next(t for t in tools if t.name == "add_item")

            # Invoke the tool
            add_tool.invoke({"name": "Widget"})

            # Verify it was tracked
            log = client.recovery_manager.log.snapshot()
            assert len(log) == 1
            record = list(log.values())[0]
            assert record.action == "add_item"
            assert record.status == ActionStatus.COMPLETED


@requires_mcp_adapters
class TestMCPToolExecutor:
    """Tests for MCPToolExecutor."""

    @pytest.mark.asyncio
    async def test_execute_async_finds_tool(self):
        """Test async execution finds and invokes tool."""
        mock_tool = MockMCPTool("add_item")
        mock_tool.ainvoke = AsyncMock(return_value='{"id": "123"}')

        mock_tools = [mock_tool]

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient"
        ) as MockClient:
            mock_client = MockMultiServerMCPClient({})
            mock_client._tools = mock_tools
            MockClient.return_value = mock_client

            client = MCPCompensationClient(
                server_config={"test": {"url": "http://localhost:8000/sse"}}
            )
            await client.connect()

            executor = MCPToolExecutor(client)
            result = await executor.execute_async("add_item", {"name": "Widget"})

            assert result == '{"id": "123"}'
            mock_tool.ainvoke.assert_called_once_with({"name": "Widget"})

    @pytest.mark.asyncio
    async def test_execute_async_tool_not_found(self):
        """Test async execution raises for unknown tool."""
        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient"
        ) as MockClient:
            mock_client = MockMultiServerMCPClient({})
            mock_client._tools = []
            MockClient.return_value = mock_client

            client = MCPCompensationClient(
                server_config={"test": {"url": "http://localhost:8000/sse"}}
            )
            await client.connect()

            executor = MCPToolExecutor(client)

            with pytest.raises(ValueError, match="Tool not found"):
                await executor.execute_async("nonexistent", {})


@requires_mcp_adapters
class TestMCPCompensationClientFallback:
    """Tests for fallback behavior when sessions not available."""

    @pytest.mark.asyncio
    async def test_fallback_to_tool_objects(self):
        """Test fallback when raw schemas not available."""
        # Tools without compensation annotations in schemas
        mock_tools = [
            MockMCPTool("add_item"),
            MockMCPTool("delete_item"),
        ]

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient"
        ) as MockClient:
            mock_client = MockMultiServerMCPClient({})
            mock_client._tools = mock_tools
            # No _sessions available
            MockClient.return_value = mock_client

            client = MCPCompensationClient(
                server_config={"test": {"url": "http://localhost:8000/sse"}}
            )
            await client.connect()

            # Should still connect successfully
            assert client.connected is True

            # No pairs discovered (fallback uses tool objects which lack annotations)
            pairs = await client.get_compensation_pairs()
            assert pairs == {}
