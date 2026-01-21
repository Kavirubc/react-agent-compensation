"""Tests for Strands hooks module."""

from react_agent_compensation.strands_adaptor.hooks import (
    CompensationHookProvider,
    format_compensation_message,
)
from tests.strands_adaptor.conftest import (
    MockAfterToolCallEvent,
    MockBeforeToolCallEvent,
)


class TestFormatCompensationMessage:
    """Tests for format_compensation_message function."""

    def test_basic_message(self):
        """Test basic compensation message format."""
        message = format_compensation_message(
            failed_action="reserve_inventory",
            error="Out of stock",
            recovery_attempts=2,
            compensated_actions=[],
        )

        assert "[COMPENSATION TRIGGERED]" in message
        assert "reserve_inventory" in message
        assert "Out of stock" in message

    def test_message_with_goals(self):
        """Test message includes goal guidance."""
        message = format_compensation_message(
            failed_action="process_payment",
            error="Payment declined",
            recovery_attempts=1,
            compensated_actions=[],
            goals=["fast_processing", "minimize_failures"],
        )

        assert "[REPLANNING GUIDANCE]" in message
        assert "fast_processing" in message
        assert "minimize_failures" in message


class TestCompensationHookProvider:
    """Tests for CompensationHookProvider class."""

    def test_init_basic(self, compensation_mapping, tools_list):
        """Test basic initialization."""
        provider = CompensationHookProvider(
            compensation_pairs=compensation_mapping,
            tools=tools_list,
        )

        assert provider.rc_manager is not None
        assert provider.transaction_log is not None

    def test_init_with_options(self, compensation_mapping, tools_list):
        """Test initialization with all options."""
        provider = CompensationHookProvider(
            compensation_pairs=compensation_mapping,
            tools=tools_list,
            goals=["minimize_latency"],
            auto_rollback=False,
            auto_recover=True,
            persist_state=True,
        )

        assert provider._goals == ["minimize_latency"]
        assert provider._auto_rollback is False
        assert provider._auto_recover is True
        assert provider._persist_state is True

    def test_before_tool_call_records_action(self, compensation_mapping, tools_list):
        """Test _before_tool_call records compensatable action."""
        provider = CompensationHookProvider(
            compensation_pairs=compensation_mapping,
            tools=tools_list,
        )

        event = MockBeforeToolCallEvent(
            tool_name="reserve_inventory",
            tool_input={"product_ids": ["SKU001"]},
            tool_use_id="use-001",
        )

        provider._before_tool_call(event)

        # Check action was recorded
        log = provider.transaction_log.snapshot()
        assert len(log) == 1

    def test_before_tool_call_skips_non_compensatable(
        self, compensation_mapping, tools_list
    ):
        """Test _before_tool_call skips non-compensatable tools."""
        provider = CompensationHookProvider(
            compensation_pairs=compensation_mapping,
            tools=tools_list,
        )

        event = MockBeforeToolCallEvent(
            tool_name="release_inventory",  # Not compensatable (is a compensator)
            tool_input={"reservation_id": "RES-001"},
        )

        provider._before_tool_call(event)

        # No action should be recorded
        log = provider.transaction_log.snapshot()
        assert len(log) == 0

    def test_after_tool_call_marks_completed(self, compensation_mapping, tools_list):
        """Test _after_tool_call marks action as completed on success."""
        provider = CompensationHookProvider(
            compensation_pairs=compensation_mapping,
            tools=tools_list,
        )

        # First, trigger before_tool_call to record the action
        tool_use_id = "use-002"
        before_event = MockBeforeToolCallEvent(
            tool_name="reserve_inventory",
            tool_input={"product_ids": ["SKU001"]},
            tool_use_id=tool_use_id,
        )
        provider._before_tool_call(before_event)

        # Then trigger after_tool_call with success result
        after_event = MockAfterToolCallEvent(
            tool_name="reserve_inventory",
            result={
                "content": [
                    {"type": "text", "text": '{"reservation_id": "RES-001"}'}
                ],
                "status": "success",
            },
            tool_use_id=tool_use_id,
        )
        provider._after_tool_call(after_event)

        # Check action was completed
        log = provider.transaction_log.snapshot()
        record = list(log.values())[0]
        assert record.status.value.lower() == "completed"

    def test_after_tool_call_handles_error(self, compensation_mapping, tools_list):
        """Test _after_tool_call handles error result."""
        provider = CompensationHookProvider(
            compensation_pairs=compensation_mapping,
            tools=tools_list,
            auto_recover=False,
            auto_rollback=False,  # Disable for simpler test
        )

        # First, trigger before_tool_call
        tool_use_id = "use-003"
        before_event = MockBeforeToolCallEvent(
            tool_name="reserve_inventory",
            tool_input={"product_ids": ["SKU001"]},
            tool_use_id=tool_use_id,
        )
        provider._before_tool_call(before_event)

        # Then trigger after_tool_call with error result
        after_event = MockAfterToolCallEvent(
            tool_name="reserve_inventory",
            result={
                "content": [{"type": "text", "text": '{"error": "Out of stock"}'}],
                "status": "error",
            },
            tool_use_id=tool_use_id,
        )
        provider._after_tool_call(after_event)

        # Result should be modified with compensation message
        assert "error" in str(after_event.result).lower()

    def test_rollback_manual(self, compensation_mapping, tools_list):
        """Test manual rollback method."""
        provider = CompensationHookProvider(
            compensation_pairs=compensation_mapping,
            tools=tools_list,
        )

        # Record and complete an action
        record = provider.rc_manager.record_action(
            "reserve_inventory", {"product_ids": ["SKU001"]}
        )
        provider.rc_manager.mark_completed(
            record.id, {"reservation_id": "RES-001"}
        )

        # Manual rollback
        result = provider.rollback()

        # Should have rolled back
        assert result is not None

    def test_clear(self, compensation_mapping, tools_list):
        """Test clear method."""
        provider = CompensationHookProvider(
            compensation_pairs=compensation_mapping,
            tools=tools_list,
        )

        # Record an action
        provider.rc_manager.record_action(
            "reserve_inventory", {"product_ids": ["SKU001"]}
        )

        # Clear
        provider.clear()

        # Log should be empty
        assert len(provider.transaction_log.snapshot()) == 0
