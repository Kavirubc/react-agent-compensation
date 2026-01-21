"""Pytest fixtures for CrewAI adaptor tests."""

import pytest
from typing import Any


class MockCrewAITool:
    """Mock CrewAI tool for testing."""

    def __init__(
        self,
        name: str,
        description: str = "",
        result: str | None = None,
        error: Exception | None = None,
    ):
        """Initialize mock tool.

        Args:
            name: Tool name
            description: Tool description
            result: Result to return (default: success JSON)
            error: Exception to raise if set
        """
        self.name = name
        self.description = description or f"Mock tool: {name}"
        self._result = result or '{"status": "ok"}'
        self._error = error
        self._calls: list[dict[str, Any]] = []

    def _run(self, **kwargs) -> str:
        """Execute the mock tool."""
        self._calls.append(kwargs)
        if self._error:
            raise self._error
        return self._result

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Get list of calls made to this tool."""
        return self._calls


class MockToolCallContext:
    """Mock CrewAI ToolCallHookContext for testing hooks."""

    def __init__(
        self,
        tool_name: str,
        tool_input: dict[str, Any] | None = None,
        tool_result: str | None = None,
        agent: Any = None,
        task: Any = None,
        crew: Any = None,
    ):
        """Initialize mock context.

        Args:
            tool_name: Name of the tool
            tool_input: Input parameters for the tool
            tool_result: Result from tool execution
            agent: Mock agent
            task: Mock task
            crew: Mock crew
        """
        self.tool_name = tool_name
        self.tool_input = tool_input or {}
        self.tool_result = tool_result
        self.agent = agent
        self.task = task
        self.crew = crew


@pytest.fixture
def mock_book_tool():
    """Fixture for a mock booking tool."""
    return MockCrewAITool(
        name="Book Flight",
        result='{"booking_id": "FL-001", "status": "confirmed"}',
    )


@pytest.fixture
def mock_cancel_tool():
    """Fixture for a mock cancellation tool."""
    return MockCrewAITool(
        name="Cancel Flight",
        result='{"cancelled": true, "booking_id": "FL-001"}',
    )


@pytest.fixture
def mock_failing_tool():
    """Fixture for a tool that returns an error."""
    return MockCrewAITool(
        name="Failing Tool",
        result='{"error": "Something went wrong"}',
    )


@pytest.fixture
def mock_exception_tool():
    """Fixture for a tool that raises an exception."""
    return MockCrewAITool(
        name="Exception Tool",
        error=ValueError("Tool execution failed"),
    )


@pytest.fixture
def compensation_mapping():
    """Fixture for basic compensation mapping."""
    return {
        "Book Flight": "Cancel Flight",
        "Book Hotel": "Cancel Hotel",
    }


@pytest.fixture
def tools_list(mock_book_tool, mock_cancel_tool):
    """Fixture for a list of tools."""
    return [mock_book_tool, mock_cancel_tool]
