"""Tests for CompensatedMCPTool."""

import json
import pytest

from react_agent_compensation.core.models import ActionStatus
from react_agent_compensation.core.recovery_manager import RecoveryManager
from react_agent_compensation.core.mcp.tools import (
    CompensatedMCPTool,
    MCPToolError,
    wrap_mcp_tools,
)


class MockMCPTool:
    """Mock MCP tool for testing."""

    def __init__(
        self,
        name: str = "test_tool",
        description: str = "Test tool",
        result: str | dict | None = None,
        error: Exception | None = None,
    ):
        self.name = name
        self.description = description
        self._result = result or {"status": "ok"}
        self._error = error
        self.invoke_calls = []
        self.ainvoke_calls = []

    def invoke(self, input: dict, config: dict | None = None) -> str:
        self.invoke_calls.append(input)
        if self._error:
            raise self._error
        if isinstance(self._result, dict):
            return json.dumps(self._result)
        return self._result

    async def ainvoke(self, input: dict, config: dict | None = None) -> str:
        self.ainvoke_calls.append(input)
        if self._error:
            raise self._error
        if isinstance(self._result, dict):
            return json.dumps(self._result)
        return self._result


class TestCompensatedMCPToolBasics:
    """Basic tests for CompensatedMCPTool."""

    def test_create_wrapped_tool(self):
        """Test wrapping an MCP tool."""
        inner = MockMCPTool(name="add_item", description="Add an item")
        manager = RecoveryManager(compensation_pairs={"add_item": "delete_item"})

        wrapped = CompensatedMCPTool(inner, manager)

        assert wrapped.name == "add_item"
        assert wrapped.description == "Add an item"
        assert wrapped.is_compensatable is True

    def test_non_compensatable_tool(self):
        """Test wrapping a tool without compensation pair."""
        inner = MockMCPTool(name="get_items")
        manager = RecoveryManager(compensation_pairs={"add_item": "delete_item"})

        wrapped = CompensatedMCPTool(inner, manager)

        assert wrapped.name == "get_items"
        assert wrapped.is_compensatable is False

    def test_metadata_access(self):
        """Test metadata property."""
        inner = MockMCPTool()
        manager = RecoveryManager(compensation_pairs={})
        metadata = {"x-action-type": "create", "x-category": "items"}

        wrapped = CompensatedMCPTool(inner, manager, metadata=metadata)

        assert wrapped.metadata == metadata
        assert wrapped.metadata["x-action-type"] == "create"


class TestCompensatedMCPToolInvoke:
    """Tests for tool invocation with tracking."""

    def test_successful_invoke_compensatable(self):
        """Test successful invocation of compensatable tool."""
        inner = MockMCPTool(
            name="add_item",
            result={"id": "123", "name": "Widget"},
        )
        manager = RecoveryManager(compensation_pairs={"add_item": "delete_item"})
        wrapped = CompensatedMCPTool(inner, manager)

        result = wrapped.invoke({"name": "Widget"})

        # Verify tool was called
        assert len(inner.invoke_calls) == 1
        assert inner.invoke_calls[0] == {"name": "Widget"}

        # Verify action was recorded and completed
        records = list(manager.log.snapshot().values())
        assert len(records) == 1
        assert records[0].action == "add_item"
        assert records[0].status == ActionStatus.COMPLETED
        assert records[0].result == {"id": "123", "name": "Widget"}

    def test_successful_invoke_non_compensatable(self):
        """Test non-compensatable tool doesn't record action."""
        inner = MockMCPTool(
            name="get_items",
            result={"items": [{"id": "1"}]},
        )
        manager = RecoveryManager(compensation_pairs={"add_item": "delete_item"})
        wrapped = CompensatedMCPTool(inner, manager)

        result = wrapped.invoke({})

        # Verify no action was recorded
        assert len(manager.log) == 0

    def test_error_result_marks_failed(self):
        """Test that error in result marks action as failed."""
        inner = MockMCPTool(
            name="add_item",
            result={"error": "Item already exists"},
        )
        manager = RecoveryManager(compensation_pairs={"add_item": "delete_item"})
        wrapped = CompensatedMCPTool(inner, manager)

        with pytest.raises(MCPToolError) as exc_info:
            wrapped.invoke({"name": "Widget"})

        assert "Item already exists" in str(exc_info.value)
        assert exc_info.value.tool_name == "add_item"

        # Verify action was marked failed
        records = list(manager.log.snapshot().values())
        assert len(records) == 1
        assert records[0].status == ActionStatus.FAILED

    def test_exception_marks_failed(self):
        """Test that exceptions mark action as failed."""
        inner = MockMCPTool(
            name="add_item",
            error=ConnectionError("Server unavailable"),
        )
        manager = RecoveryManager(compensation_pairs={"add_item": "delete_item"})
        wrapped = CompensatedMCPTool(inner, manager)

        with pytest.raises(ConnectionError):
            wrapped.invoke({"name": "Widget"})

        # Verify action was marked failed
        records = list(manager.log.snapshot().values())
        assert len(records) == 1
        assert records[0].status == ActionStatus.FAILED

    def test_status_error_detected(self):
        """Test that status: error is detected."""
        inner = MockMCPTool(
            name="add_item",
            result={"status": "error", "message": "Validation failed"},
        )
        manager = RecoveryManager(compensation_pairs={"add_item": "delete_item"})
        wrapped = CompensatedMCPTool(inner, manager)

        with pytest.raises(MCPToolError):
            wrapped.invoke({"name": "Widget"})


class TestCompensatedMCPToolAsync:
    """Tests for async invocation."""

    @pytest.mark.asyncio
    async def test_async_invoke_compensatable(self):
        """Test async invocation of compensatable tool."""
        inner = MockMCPTool(
            name="add_item",
            result={"id": "456", "name": "Gadget"},
        )
        manager = RecoveryManager(compensation_pairs={"add_item": "delete_item"})
        wrapped = CompensatedMCPTool(inner, manager)

        result = await wrapped.ainvoke({"name": "Gadget"})

        # Verify async was called
        assert len(inner.ainvoke_calls) == 1

        # Verify action was recorded
        records = list(manager.log.snapshot().values())
        assert len(records) == 1
        assert records[0].status == ActionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_async_error_handling(self):
        """Test async error handling."""
        inner = MockMCPTool(
            name="add_item",
            result={"error": "Async error"},
        )
        manager = RecoveryManager(compensation_pairs={"add_item": "delete_item"})
        wrapped = CompensatedMCPTool(inner, manager)

        with pytest.raises(MCPToolError):
            await wrapped.ainvoke({"name": "Widget"})

        records = list(manager.log.snapshot().values())
        assert records[0].status == ActionStatus.FAILED


class TestErrorDetection:
    """Tests for error result detection."""

    def test_detect_error_key_dict(self):
        """Test detection of error key in dict."""
        inner = MockMCPTool()
        manager = RecoveryManager(compensation_pairs={})
        wrapped = CompensatedMCPTool(inner, manager)

        assert wrapped._is_error_result({"error": "something"}) is True
        assert wrapped._is_error_result({"status": "ok"}) is False

    def test_detect_status_error(self):
        """Test detection of status: error."""
        inner = MockMCPTool()
        manager = RecoveryManager(compensation_pairs={})
        wrapped = CompensatedMCPTool(inner, manager)

        assert wrapped._is_error_result({"status": "error"}) is True
        assert wrapped._is_error_result({"status": "success"}) is False

    def test_detect_error_in_json_string(self):
        """Test detection of error in JSON string."""
        inner = MockMCPTool()
        manager = RecoveryManager(compensation_pairs={})
        wrapped = CompensatedMCPTool(inner, manager)

        assert wrapped._is_error_result('{"error": "bad"}') is True
        assert wrapped._is_error_result('{"id": "123"}') is False

    def test_non_json_string_not_error(self):
        """Test non-JSON strings are not errors."""
        inner = MockMCPTool()
        manager = RecoveryManager(compensation_pairs={})
        wrapped = CompensatedMCPTool(inner, manager)

        assert wrapped._is_error_result("plain text") is False
        assert wrapped._is_error_result("error: something") is False


class TestResultParsing:
    """Tests for result parsing."""

    def test_parse_dict_result(self):
        """Test parsing dict results."""
        inner = MockMCPTool()
        manager = RecoveryManager(compensation_pairs={})
        wrapped = CompensatedMCPTool(inner, manager)

        result = wrapped._parse_result({"id": "123"})
        assert result == {"id": "123"}

    def test_parse_json_string(self):
        """Test parsing JSON string results."""
        inner = MockMCPTool()
        manager = RecoveryManager(compensation_pairs={})
        wrapped = CompensatedMCPTool(inner, manager)

        result = wrapped._parse_result('{"id": "123"}')
        assert result == {"id": "123"}

    def test_parse_plain_string(self):
        """Test parsing plain string results."""
        inner = MockMCPTool()
        manager = RecoveryManager(compensation_pairs={})
        wrapped = CompensatedMCPTool(inner, manager)

        result = wrapped._parse_result("plain text")
        assert result == {"raw": "plain text"}


class TestWrapMCPTools:
    """Tests for wrap_mcp_tools function."""

    def test_wrap_multiple_tools(self):
        """Test wrapping multiple tools."""
        tools = [
            MockMCPTool(name="add_item"),
            MockMCPTool(name="delete_item"),
            MockMCPTool(name="get_items"),
        ]
        manager = RecoveryManager(compensation_pairs={"add_item": "delete_item"})

        wrapped = wrap_mcp_tools(tools, manager)

        assert len(wrapped) == 3
        assert all(isinstance(t, CompensatedMCPTool) for t in wrapped)
        assert wrapped[0].name == "add_item"
        assert wrapped[0].is_compensatable is True
        assert wrapped[2].name == "get_items"
        assert wrapped[2].is_compensatable is False

    def test_wrap_with_metadata(self):
        """Test wrapping with per-tool metadata."""
        tools = [
            MockMCPTool(name="add_item"),
            MockMCPTool(name="get_items"),
        ]
        manager = RecoveryManager(compensation_pairs={})
        metadata = {
            "add_item": {"x-action-type": "create"},
            "get_items": {"x-action-type": "read"},
        }

        wrapped = wrap_mcp_tools(tools, manager, metadata)

        assert wrapped[0].metadata == {"x-action-type": "create"}
        assert wrapped[1].metadata == {"x-action-type": "read"}


class TestMCPToolError:
    """Tests for MCPToolError exception."""

    def test_create_error(self):
        """Test creating an MCPToolError."""
        error = MCPToolError(
            "Item not found",
            tool_name="get_item",
            result={"error": "Item not found"},
        )

        assert str(error) == "Item not found"
        assert error.tool_name == "get_item"
        assert error.result == {"error": "Item not found"}

    def test_error_minimal(self):
        """Test creating error with minimal info."""
        error = MCPToolError("Something went wrong", tool_name="my_tool")

        assert str(error) == "Something went wrong"
        assert error.tool_name == "my_tool"
        assert error.result is None
