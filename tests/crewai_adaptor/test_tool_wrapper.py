"""Tests for CrewAI tool_wrapper module."""

import pytest

from react_agent_compensation.crewai_adaptor.middleware import CrewAICompensationMiddleware
from react_agent_compensation.crewai_adaptor.tool_wrapper import (
    format_compensation_message,
    wrap_tool_with_compensation,
)


class TestFormatCompensationMessage:
    """Tests for format_compensation_message function."""

    def test_basic_message(self):
        """Test basic compensation message format."""
        message = format_compensation_message(
            failed_action="Book Flight",
            error="Connection timeout",
            recovery_attempts=2,
            compensated_actions=[],
        )

        assert "[COMPENSATION TRIGGERED]" in message
        assert "Book Flight" in message
        assert "Connection timeout" in message
        assert "2 retry attempt(s)" in message

    def test_message_with_rollback_details(self):
        """Test message includes rollback details."""
        rollback_details = [
            {
                "action": "Book Hotel",
                "compensator": "Cancel Hotel",
                "params": {"reservation_id": "HT-001"},
            }
        ]

        message = format_compensation_message(
            failed_action="Process Payment",
            error="Payment failed",
            recovery_attempts=1,
            compensated_actions=["rec-1"],
            rollback_details=rollback_details,
        )

        assert "[ROLLBACK EXECUTED" in message
        assert "Book Hotel" in message
        assert "Cancel Hotel" in message
        assert "CANCELLED" in message

    def test_message_with_goals(self):
        """Test message includes goal guidance."""
        message = format_compensation_message(
            failed_action="Book Flight",
            error="No availability",
            recovery_attempts=1,
            compensated_actions=[],
            goals=["minimize_cost", "prefer_direct_flights"],
        )

        assert "[REPLANNING GUIDANCE]" in message
        assert "minimize_cost" in message
        assert "prefer_direct_flights" in message

    def test_message_with_failure_context(self):
        """Test message includes failure context summary."""
        message = format_compensation_message(
            failed_action="API Call",
            error="Rate limited",
            recovery_attempts=3,
            compensated_actions=[],
            failure_context_summary="Previous failures: API call failed 3 times",
        )

        assert "Previous failures: API call failed 3 times" in message


class TestWrapToolWithCompensation:
    """Tests for wrap_tool_with_compensation function."""

    def test_wrap_basic_tool(self, mock_book_tool, compensation_mapping, tools_list):
        """Test wrapping a basic tool."""
        middleware = CrewAICompensationMiddleware(
            compensation_mapping=compensation_mapping,
            tools=tools_list,
        )

        wrapped = wrap_tool_with_compensation(mock_book_tool, middleware)

        # Wrapped tool should be callable
        assert callable(wrapped) or hasattr(wrapped, "_run")

    def test_wrapped_tool_records_action(
        self, mock_book_tool, compensation_mapping, tools_list
    ):
        """Test wrapped tool records action before execution."""
        middleware = CrewAICompensationMiddleware(
            compensation_mapping=compensation_mapping,
            tools=tools_list,
        )

        wrapped = wrap_tool_with_compensation(mock_book_tool, middleware)

        # Execute wrapped tool
        if hasattr(wrapped, "_run"):
            wrapped._run(destination="NYC")
        else:
            wrapped(destination="NYC")

        # Check action was recorded
        log = middleware.transaction_log.snapshot()
        assert len(log) == 1

    def test_wrapped_tool_marks_completed(
        self, mock_book_tool, compensation_mapping, tools_list
    ):
        """Test wrapped tool marks action as completed on success."""
        middleware = CrewAICompensationMiddleware(
            compensation_mapping=compensation_mapping,
            tools=tools_list,
        )

        wrapped = wrap_tool_with_compensation(mock_book_tool, middleware)

        # Execute wrapped tool
        if hasattr(wrapped, "_run"):
            wrapped._run(destination="NYC")
        else:
            wrapped(destination="NYC")

        # Check action was completed
        log = middleware.transaction_log.snapshot()
        record = list(log.values())[0]
        assert record.status.value.lower() == "completed"

    def test_wrapped_tool_handles_error_result(
        self, mock_failing_tool, compensation_mapping
    ):
        """Test wrapped tool handles error in result."""
        tools = [mock_failing_tool]
        middleware = CrewAICompensationMiddleware(
            compensation_mapping={"Failing Tool": "Cancel Tool"},
            tools=tools,
        )

        wrapped = wrap_tool_with_compensation(
            mock_failing_tool,
            middleware,
            auto_rollback=False,  # Disable rollback for simpler test
            auto_recover=False,
        )

        # Execute wrapped tool - should return compensation message
        if hasattr(wrapped, "_run"):
            result = wrapped._run()
        else:
            result = wrapped()

        # Result should contain compensation message
        assert "[COMPENSATION TRIGGERED]" in result

    def test_wrapped_non_compensatable_tool(
        self, mock_cancel_tool, compensation_mapping, tools_list
    ):
        """Test wrapped non-compensatable tool doesn't record action."""
        middleware = CrewAICompensationMiddleware(
            compensation_mapping=compensation_mapping,
            tools=tools_list,
        )

        wrapped = wrap_tool_with_compensation(mock_cancel_tool, middleware)

        # Execute wrapped tool (Cancel Flight is not compensatable)
        if hasattr(wrapped, "_run"):
            wrapped._run(booking_id="FL-001")
        else:
            wrapped(booking_id="FL-001")

        # No action should be recorded for non-compensatable tool
        log = middleware.transaction_log.snapshot()
        assert len(log) == 0
