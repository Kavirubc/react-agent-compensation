"""Tests for Strands state module."""

from react_agent_compensation.core.models import FailureContext
from react_agent_compensation.core.transaction_log import TransactionLog
from react_agent_compensation.strands_adaptor.state import (
    COMPENSATION_LOG_KEY,
    FAILURE_CONTEXT_KEY,
    StrandsStateSync,
    get_compensation_log,
    get_failure_context,
    sync_compensation_log,
    sync_failure_context,
)


class TestStrandsStateSync:
    """Tests for StrandsStateSync class."""

    def test_save_and_load_transaction_log(self):
        """Test saving and loading transaction log."""
        sync = StrandsStateSync()
        invocation_state: dict = {}

        # Create a log with an action
        log = TransactionLog()
        from react_agent_compensation.core.models import ActionRecord, ActionStatus

        record = ActionRecord(
            action="reserve_inventory",
            params={"product_ids": ["SKU001"]},
            status=ActionStatus.COMPLETED,
            result={"reservation_id": "RES-001"},
        )
        log.add(record)

        # Save to invocation_state
        sync.save_transaction_log(invocation_state, log)

        # Load back
        loaded_log = sync.load_transaction_log(invocation_state)

        # Should have the same record
        assert len(loaded_log.snapshot()) == 1

    def test_save_and_load_failure_context(self):
        """Test saving and loading failure context."""
        sync = StrandsStateSync()
        invocation_state: dict = {}

        # Create failure context with an attempt
        ctx = FailureContext()
        ctx.record_attempt(
            action="process_payment",
            params={"amount": 100},
            error="Payment declined",
            is_permanent=False,
        )

        # Save to invocation_state
        sync.save_failure_context(invocation_state, ctx)

        # Load back
        loaded_ctx = sync.load_failure_context(invocation_state)

        # Should have the same failure
        assert len(loaded_ctx.attempts) == 1

    def test_load_empty_state(self):
        """Test loading from empty state."""
        sync = StrandsStateSync()
        invocation_state: dict = {}

        log = sync.load_transaction_log(invocation_state)
        ctx = sync.load_failure_context(invocation_state)

        assert log is not None
        assert ctx is not None


class TestStateFunctions:
    """Tests for standalone state functions."""

    def test_get_compensation_log_found(self):
        """Test getting log when it exists."""
        state = {
            COMPENSATION_LOG_KEY: {"records": {}, "global_counter": 0}
        }

        log = get_compensation_log(state)

        assert log is not None
        assert isinstance(log, TransactionLog)

    def test_get_compensation_log_not_found(self):
        """Test getting log when it doesn't exist."""
        state: dict = {}

        log = get_compensation_log(state)

        assert log is None

    def test_sync_compensation_log(self):
        """Test syncing log to state."""
        state: dict = {}
        log = TransactionLog()

        sync_compensation_log(state, log)

        assert COMPENSATION_LOG_KEY in state

    def test_get_failure_context_found(self):
        """Test getting failure context when it exists."""
        state = {
            FAILURE_CONTEXT_KEY: {"failed_attempts": [], "permanent_failures": []}
        }

        ctx = get_failure_context(state)

        assert ctx is not None
        assert isinstance(ctx, FailureContext)

    def test_get_failure_context_not_found(self):
        """Test getting failure context when it doesn't exist."""
        state: dict = {}

        ctx = get_failure_context(state)

        assert ctx is None

    def test_sync_failure_context(self):
        """Test syncing failure context to state."""
        state: dict = {}
        ctx = FailureContext()

        sync_failure_context(state, ctx)

        assert FAILURE_CONTEXT_KEY in state
