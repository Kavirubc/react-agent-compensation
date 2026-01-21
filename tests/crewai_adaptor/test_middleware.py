"""Tests for CrewAI middleware module."""

import pytest

from react_agent_compensation.crewai_adaptor.middleware import CrewAICompensationMiddleware
from react_agent_compensation.core.config import RetryPolicy


class TestCrewAICompensationMiddleware:
    """Tests for CrewAICompensationMiddleware class."""

    def test_init_basic(self, compensation_mapping, tools_list):
        """Test basic initialization."""
        middleware = CrewAICompensationMiddleware(
            compensation_mapping=compensation_mapping,
            tools=tools_list,
        )

        assert middleware.compensation_mapping == compensation_mapping
        assert middleware.rc_manager is not None
        assert middleware.transaction_log is not None

    def test_init_with_options(self, compensation_mapping, tools_list):
        """Test initialization with all options."""
        policy = RetryPolicy(max_retries=3)

        middleware = CrewAICompensationMiddleware(
            compensation_mapping=compensation_mapping,
            tools=tools_list,
            retry_policy=policy,
            crew_id="crew-1",
            agent_id="agent-1",
            goals=["minimize_cost"],
        )

        assert middleware.crew_id == "crew-1"
        assert middleware.agent_id == "agent-1"
        assert middleware.goals == ["minimize_cost"]

    def test_is_compensatable_true(self, compensation_mapping, tools_list):
        """Test is_compensatable for registered tool."""
        middleware = CrewAICompensationMiddleware(
            compensation_mapping=compensation_mapping,
            tools=tools_list,
        )

        assert middleware.is_compensatable("Book Flight") is True

    def test_is_compensatable_false(self, compensation_mapping, tools_list):
        """Test is_compensatable for unregistered tool."""
        middleware = CrewAICompensationMiddleware(
            compensation_mapping=compensation_mapping,
            tools=tools_list,
        )

        assert middleware.is_compensatable("Unknown Tool") is False

    def test_get_tool(self, compensation_mapping, tools_list, mock_book_tool):
        """Test get_tool returns correct tool."""
        middleware = CrewAICompensationMiddleware(
            compensation_mapping=compensation_mapping,
            tools=tools_list,
        )

        tool = middleware.get_tool("Book Flight")

        assert tool is mock_book_tool

    def test_get_tool_not_found(self, compensation_mapping, tools_list):
        """Test get_tool returns None for unknown tool."""
        middleware = CrewAICompensationMiddleware(
            compensation_mapping=compensation_mapping,
            tools=tools_list,
        )

        tool = middleware.get_tool("Unknown Tool")

        assert tool is None

    def test_add_tool(self, compensation_mapping, tools_list, mock_cancel_tool):
        """Test adding a tool to the cache."""
        middleware = CrewAICompensationMiddleware(
            compensation_mapping=compensation_mapping,
            tools=[],
        )

        middleware.add_tool(mock_cancel_tool)

        assert middleware.get_tool("Cancel Flight") is mock_cancel_tool

    def test_clear(self, compensation_mapping, tools_list):
        """Test clearing the transaction log."""
        middleware = CrewAICompensationMiddleware(
            compensation_mapping=compensation_mapping,
            tools=tools_list,
        )

        # Record an action first
        middleware.rc_manager.record_action("Book Flight", {"destination": "NYC"})

        # Clear
        middleware.clear()

        # Log should be empty
        assert len(middleware.transaction_log.snapshot()) == 0

    def test_get_failure_summary_empty(self, compensation_mapping, tools_list):
        """Test get_failure_summary with no failures."""
        middleware = CrewAICompensationMiddleware(
            compensation_mapping=compensation_mapping,
            tools=tools_list,
        )

        summary = middleware.get_failure_summary()

        # Should be empty or indicate no failures
        assert isinstance(summary, str)

    def test_shared_log(self, compensation_mapping, tools_list):
        """Test using a shared transaction log."""
        from react_agent_compensation.core.transaction_log import TransactionLog

        shared_log = TransactionLog()

        middleware = CrewAICompensationMiddleware(
            compensation_mapping=compensation_mapping,
            tools=tools_list,
            shared_log=shared_log,
        )

        # Should use the shared log
        assert middleware.transaction_log is shared_log
