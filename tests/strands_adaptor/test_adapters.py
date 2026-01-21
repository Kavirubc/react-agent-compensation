"""Tests for Strands adapters module."""

import pytest

from react_agent_compensation.strands_adaptor.adapters import (
    SimpleActionResult,
    StrandsActionResult,
    StrandsToolExecutor,
    build_tools_cache,
)


class TestStrandsActionResult:
    """Tests for StrandsActionResult class."""

    def test_parse_strands_format(self):
        """Test parsing Strands format result."""
        result = StrandsActionResult(
            result={
                "content": [{"type": "text", "text": '{"reservation_id": "RES-001"}'}],
                "status": "success",
            },
            tool_name="reserve_inventory",
        )

        assert result.content == {"reservation_id": "RES-001"}
        assert result.name == "reserve_inventory"
        # status can be "success" from the raw result or None if not error
        assert result.status in (None, "success")

    def test_parse_string_json_result(self):
        """Test parsing JSON string result."""
        result = StrandsActionResult(
            result='{"payment_id": "PAY-001"}',
            tool_name="process_payment",
        )

        assert result.content == {"payment_id": "PAY-001"}
        assert result.name == "process_payment"

    def test_parse_plain_string_result(self):
        """Test parsing plain string result."""
        result = StrandsActionResult(
            result="Operation completed",
            tool_name="simple_tool",
        )

        assert result.content == "Operation completed"

    def test_detect_error_from_status(self):
        """Test error detection from status field."""
        result = StrandsActionResult(
            result={
                "content": [{"type": "text", "text": "Error message"}],
                "status": "error",
            },
            tool_name="failing_tool",
        )

        assert result.status == "error"

    def test_detect_error_from_content(self):
        """Test error detection from content with error field."""
        result = StrandsActionResult(
            result={
                "content": [{"type": "text", "text": '{"error": "Payment failed"}'}],
            },
            tool_name="process_payment",
        )

        assert result.status == "error"

    def test_raw_property(self):
        """Test raw property returns original value."""
        original = {"content": [{"type": "text", "text": "test"}]}
        result = StrandsActionResult(result=original, tool_name="Test")

        assert result.raw == original

    def test_action_id_generated(self):
        """Test action_id is generated if not provided."""
        result = StrandsActionResult(result="test", tool_name="Test")

        assert result.action_id is not None
        assert len(result.action_id) > 0


class TestStrandsToolExecutor:
    """Tests for StrandsToolExecutor class."""

    def test_execute_callable_tool(self, mock_reserve_tool):
        """Test executing a callable tool."""
        executor = StrandsToolExecutor({"reserve_inventory": mock_reserve_tool})

        result = executor.execute("reserve_inventory", {"product_ids": ["SKU001"]})

        assert result.name == "reserve_inventory"
        assert "reservation_id" in result.content

    def test_execute_tool_not_found(self):
        """Test executing a non-existent tool."""
        executor = StrandsToolExecutor({})

        result = executor.execute("unknown_tool", {})

        assert result.status == "error"
        assert "not found" in str(result.raw)

    def test_execute_tool_raises_exception(self):
        """Test executing a tool that raises an exception."""

        def failing_tool(**kwargs):
            raise ValueError("Tool failed")

        failing_tool.name = "failing"

        executor = StrandsToolExecutor({"failing": failing_tool})

        result = executor.execute("failing", {})

        assert result.status == "error"


class TestSimpleActionResult:
    """Tests for SimpleActionResult class."""

    def test_properties(self):
        """Test all properties."""
        result = SimpleActionResult(
            content={"key": "value"},
            status="success",
            name="Test Tool",
            action_id="test-123",
        )

        assert result.content == {"key": "value"}
        assert result.status == "success"
        assert result.name == "Test Tool"
        assert result.action_id == "test-123"


class TestBuildToolsCache:
    """Tests for build_tools_cache function."""

    def test_build_from_tools_with_name(self, mock_reserve_tool, mock_release_tool):
        """Test building cache from tools with name attribute."""
        tools = [mock_reserve_tool, mock_release_tool]

        cache = build_tools_cache(tools)

        assert "reserve_inventory" in cache
        assert "release_inventory" in cache

    def test_build_from_empty_list(self):
        """Test building cache from empty list."""
        cache = build_tools_cache([])

        assert cache == {}

    def test_build_from_none(self):
        """Test building cache from None."""
        cache = build_tools_cache(None)

        assert cache == {}
