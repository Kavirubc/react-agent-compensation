"""Pytest fixtures for Strands adaptor tests."""

import pytest
from typing import Any


class MockStrandsTool:
    """Mock Strands tool for testing."""

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
            result: Result to return
            error: Exception to raise if set
        """
        self.name = name
        self.description = description or f"Mock tool: {name}"
        self._result = result or '{"status": "ok"}'
        self._error = error
        self._calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs) -> str:
        """Execute the mock tool."""
        self._calls.append(kwargs)
        if self._error:
            raise self._error
        return self._result

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Get list of calls made to this tool."""
        return self._calls


class MockBeforeToolCallEvent:
    """Mock Strands BeforeToolCallEvent for testing."""

    def __init__(
        self,
        tool_name: str,
        tool_input: dict[str, Any] | None = None,
        tool_use_id: str = "test-123",
    ):
        """Initialize mock event.

        Args:
            tool_name: Name of the tool
            tool_input: Input parameters
            tool_use_id: Unique tool use ID
        """
        self.tool_use = {
            "name": tool_name,
            "input": tool_input or {},
            "id": tool_use_id,
        }
        self.selected_tool = None
        self.cancel_tool = None
        self.invocation_state: dict[str, Any] = {}


class MockAfterToolCallEvent:
    """Mock Strands AfterToolCallEvent for testing."""

    def __init__(
        self,
        tool_name: str,
        result: dict[str, Any] | None = None,
        exception: Exception | None = None,
        tool_use_id: str = "test-123",
    ):
        """Initialize mock event.

        Args:
            tool_name: Name of the tool
            result: Tool result in Strands format
            exception: Exception if tool raised one
            tool_use_id: Unique tool use ID
        """
        self.tool_use = {
            "name": tool_name,
            "input": {},
            "id": tool_use_id,
        }
        self.result = result or {
            "content": [{"type": "text", "text": '{"status": "ok"}'}],
            "status": "success",
        }
        self.exception = exception
        self.invocation_state: dict[str, Any] = {}


@pytest.fixture
def mock_reserve_tool():
    """Fixture for a mock inventory reservation tool."""
    return MockStrandsTool(
        name="reserve_inventory",
        result='{"reservation_id": "RES-001", "status": "reserved"}',
    )


@pytest.fixture
def mock_release_tool():
    """Fixture for a mock inventory release tool."""
    return MockStrandsTool(
        name="release_inventory",
        result='{"released": true, "reservation_id": "RES-001"}',
    )


@pytest.fixture
def mock_payment_tool():
    """Fixture for a mock payment tool."""
    return MockStrandsTool(
        name="process_payment",
        result='{"payment_id": "PAY-001", "status": "completed"}',
    )


@pytest.fixture
def mock_failing_tool():
    """Fixture for a tool that returns an error."""
    return MockStrandsTool(
        name="failing_tool",
        result='{"error": "Operation failed"}',
    )


@pytest.fixture
def compensation_mapping():
    """Fixture for basic compensation mapping."""
    return {
        "reserve_inventory": "release_inventory",
        "process_payment": "refund_payment",
    }


@pytest.fixture
def tools_list(mock_reserve_tool, mock_release_tool, mock_payment_tool):
    """Fixture for a list of tools."""
    return [mock_reserve_tool, mock_release_tool, mock_payment_tool]
