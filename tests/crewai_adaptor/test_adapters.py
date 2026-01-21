"""Tests for CrewAI adapters module."""

import pytest

from react_agent_compensation.crewai_adaptor.adapters import (
    CrewAIActionResult,
    CrewAIToolExecutor,
    SimpleActionResult,
    build_tools_cache,
)


class TestCrewAIActionResult:
    """Tests for CrewAIActionResult class."""

    def test_parse_json_result(self):
        """Test parsing JSON string result."""
        result = CrewAIActionResult(
            result='{"booking_id": "FL-001", "status": "confirmed"}',
            tool_name="Book Flight",
        )

        assert result.content == {"booking_id": "FL-001", "status": "confirmed"}
        assert result.name == "Book Flight"
        assert result.status is None

    def test_parse_plain_string_result(self):
        """Test parsing plain string result."""
        result = CrewAIActionResult(
            result="Booking confirmed",
            tool_name="Book Flight",
        )

        assert result.content == "Booking confirmed"
        assert result.name == "Book Flight"

    def test_detect_error_from_dict(self):
        """Test error detection from dict content."""
        result = CrewAIActionResult(
            result='{"error": "Payment failed"}',
            tool_name="Process Payment",
        )

        assert result.status == "error"

    def test_detect_error_from_string(self):
        """Test error detection from string content."""
        result = CrewAIActionResult(
            result="Error: connection failed",
            tool_name="API Call",
        )

        assert result.status == "error"

    def test_raw_property(self):
        """Test raw property returns original string."""
        original = '{"booking_id": "FL-001"}'
        result = CrewAIActionResult(result=original, tool_name="Test")

        assert result.raw == original

    def test_action_id_generated(self):
        """Test action_id is generated if not provided."""
        result = CrewAIActionResult(result="test", tool_name="Test")

        assert result.action_id is not None
        assert len(result.action_id) > 0

    def test_action_id_provided(self):
        """Test action_id uses provided value."""
        result = CrewAIActionResult(
            result="test",
            tool_name="Test",
            action_id="custom-id",
        )

        assert result.action_id == "custom-id"


class TestCrewAIToolExecutor:
    """Tests for CrewAIToolExecutor class."""

    def test_execute_with_run_method(self, mock_book_tool):
        """Test executing a tool with _run method."""
        executor = CrewAIToolExecutor({"Book Flight": mock_book_tool})

        result = executor.execute("Book Flight", {"destination": "NYC"})

        assert result.name == "Book Flight"
        assert result.content["booking_id"] == "FL-001"
        assert mock_book_tool.calls == [{"destination": "NYC"}]

    def test_execute_tool_not_found(self):
        """Test executing a non-existent tool."""
        executor = CrewAIToolExecutor({})

        result = executor.execute("Unknown Tool", {})

        assert result.status == "error"
        assert "not found" in result.content.lower()

    def test_execute_tool_raises_exception(self, mock_exception_tool):
        """Test executing a tool that raises an exception."""
        executor = CrewAIToolExecutor({"Exception Tool": mock_exception_tool})

        result = executor.execute("Exception Tool", {})

        assert result.status == "error"
        assert "Error:" in result.content


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

    def test_action_id_generated(self):
        """Test action_id is generated if not provided."""
        result = SimpleActionResult(content="test")

        assert result.action_id is not None


class TestBuildToolsCache:
    """Tests for build_tools_cache function."""

    def test_build_from_tools_with_name(self, mock_book_tool, mock_cancel_tool):
        """Test building cache from tools with name attribute."""
        tools = [mock_book_tool, mock_cancel_tool]

        cache = build_tools_cache(tools)

        assert "Book Flight" in cache
        assert "Cancel Flight" in cache
        assert cache["Book Flight"] is mock_book_tool

    def test_build_from_empty_list(self):
        """Test building cache from empty list."""
        cache = build_tools_cache([])

        assert cache == {}

    def test_build_from_none(self):
        """Test building cache from None."""
        cache = build_tools_cache(None)

        assert cache == {}

    def test_build_from_callable_with_name(self):
        """Test building cache from callable with __name__."""
        def my_tool():
            pass

        cache = build_tools_cache([my_tool])

        assert "my_tool" in cache
