"""Tests for RecoveryManager."""

import pytest

from react_agent_compensation.core.config import RetryPolicy
from react_agent_compensation.core.exceptions import CriticalFailure, ExtractionError, RollbackFailure
from react_agent_compensation.core.models import ActionStatus
from react_agent_compensation.core.recovery_manager import RecoveryManager, RecoveryResult, RollbackResult


class MockExecutor:
    """Mock action executor for testing."""

    def __init__(self, results: dict | None = None, errors: dict | None = None):
        self.results = results or {}
        self.errors = errors or {}
        self.call_history = []

    def execute(self, action: str, params: dict):
        self.call_history.append((action, params))
        if action in self.errors:
            raise self.errors[action]
        return self.results.get(action, {"status": "ok"})


class TestRecoveryManagerBasics:
    """Basic tests for RecoveryManager."""

    def test_create_manager(self):
        """Test creating a manager."""
        manager = RecoveryManager(
            compensation_pairs={"book_flight": "cancel_flight"},
        )

        assert manager.compensation_pairs == {"book_flight": "cancel_flight"}
        assert manager.is_compensatable("book_flight") is True
        assert manager.is_compensatable("unknown") is False

    def test_record_action(self):
        """Test recording an action."""
        manager = RecoveryManager(
            compensation_pairs={"book": "cancel"},
        )

        record = manager.record_action("book", {"dest": "NYC"})

        assert record.action == "book"
        assert record.params == {"dest": "NYC"}
        assert record.status == ActionStatus.PENDING
        assert record.compensator == "cancel"
        assert record.id in manager.log

    def test_mark_completed(self):
        """Test marking an action as completed."""
        manager = RecoveryManager(compensation_pairs={})
        record = manager.record_action("test", {})

        manager.mark_completed(record.id, result={"id": "123"})

        updated = manager.log.get(record.id)
        assert updated.status == ActionStatus.COMPLETED
        assert updated.result == {"id": "123"}

    def test_mark_failed(self):
        """Test marking an action as failed."""
        manager = RecoveryManager(compensation_pairs={})
        record = manager.record_action("test", {})

        manager.mark_failed(record.id, error="Connection refused")

        updated = manager.log.get(record.id)
        assert updated.status == ActionStatus.FAILED
        assert updated.result == {"error": "Connection refused"}

    def test_clear(self):
        """Test clearing the log."""
        manager = RecoveryManager(compensation_pairs={})
        manager.record_action("test1", {})
        manager.record_action("test2", {})

        assert len(manager.log) == 2
        manager.clear()
        assert len(manager.log) == 0


class TestRecoveryManagerDependencies:
    """Tests for dependency inference."""

    def test_infers_dependency(self):
        """Test that dependencies are inferred from data flow."""
        manager = RecoveryManager(
            compensation_pairs={"book": "cancel", "pay": "refund"},
            infer_dependencies=True,
        )

        # First action returns booking_id
        rec1 = manager.record_action("book", {"dest": "NYC"})
        manager.mark_completed(rec1.id, result={"booking_id": "ABC123"})

        # Second action uses the booking_id
        rec2 = manager.record_action("pay", {"booking_id": "ABC123", "amount": 100})

        assert rec1.id in rec2.depends_on

    def test_no_dependency_when_disabled(self):
        """Test dependencies are not inferred when disabled."""
        manager = RecoveryManager(
            compensation_pairs={"book": "cancel", "pay": "refund"},
            infer_dependencies=False,
        )

        rec1 = manager.record_action("book", {"dest": "NYC"})
        manager.mark_completed(rec1.id, result={"booking_id": "ABC123"})

        rec2 = manager.record_action("pay", {"booking_id": "ABC123"})

        assert rec1.id not in rec2.depends_on


class TestRecoveryManagerAddRemove:
    """Tests for adding/removing compensation pairs."""

    def test_add_pair(self):
        """Test adding a compensation pair."""
        manager = RecoveryManager(compensation_pairs={})

        manager.add_compensation_pair("book", "cancel")

        assert manager.is_compensatable("book")

    def test_remove_pair(self):
        """Test removing a compensation pair."""
        manager = RecoveryManager(
            compensation_pairs={"book": "cancel"},
        )

        result = manager.remove_compensation_pair("book")

        assert result is True
        assert manager.is_compensatable("book") is False

    def test_remove_nonexistent(self):
        """Test removing non-existent pair."""
        manager = RecoveryManager(compensation_pairs={})

        result = manager.remove_compensation_pair("unknown")

        assert result is False


class TestRecoveryManagerRetry:
    """Tests for retry logic."""

    def test_successful_retry(self):
        """Test successful retry after failure."""
        executor = MockExecutor(
            results={"book": {"booking_id": "123"}},
        )
        manager = RecoveryManager(
            compensation_pairs={"book": "cancel"},
            retry_policy=RetryPolicy(max_retries=3, initial_delay=0.01),
            action_executor=executor,
        )

        record = manager.record_action("book", {"dest": "NYC"})
        # Use a retryable error pattern (contains "connection")
        result = manager.recover(record.id, "connection error")

        assert result.success is True
        assert result.action_taken == "retry"
        assert result.result == {"booking_id": "123"}

    def test_retry_exhausted(self):
        """Test when all retries are exhausted."""
        executor = MockExecutor(
            errors={"book": ConnectionError("failed")},
        )
        manager = RecoveryManager(
            compensation_pairs={"book": "cancel"},
            retry_policy=RetryPolicy(max_retries=2, initial_delay=0.01),
            action_executor=executor,
        )

        record = manager.record_action("book", {"dest": "NYC"})
        result = manager.recover(record.id, ConnectionError("failed"))

        assert result.success is False
        assert result.action_taken == "failed"

    def test_alternative_on_failure(self):
        """Test trying alternatives when retry fails."""
        executor = MockExecutor(
            errors={"book_primary": ConnectionError("failed")},
            results={"book_backup": {"booking_id": "456"}},
        )
        manager = RecoveryManager(
            compensation_pairs={"book_primary": "cancel", "book_backup": "cancel"},
            alternative_map={"book_primary": ["book_backup"]},
            retry_policy=RetryPolicy(max_retries=1, initial_delay=0.01),
            action_executor=executor,
        )

        record = manager.record_action("book_primary", {"dest": "NYC"})
        result = manager.recover(record.id, ConnectionError("failed"))

        assert result.success is True
        assert result.action_taken == "alternative"

    def test_nonexistent_record(self):
        """Test recover with nonexistent record."""
        manager = RecoveryManager(compensation_pairs={})

        result = manager.recover("nonexistent", "error")

        assert result.success is False
        assert "not found" in result.error


class TestRecoveryManagerRollback:
    """Tests for rollback logic."""

    def test_empty_rollback(self):
        """Test rollback with empty log."""
        manager = RecoveryManager(compensation_pairs={})

        result = manager.rollback()

        assert result.success is True
        assert result.compensated == []
        assert "No actions" in result.message

    def test_simple_rollback(self):
        """Test simple rollback with one action."""
        executor = MockExecutor(
            results={"cancel": {"status": "cancelled"}},
        )
        manager = RecoveryManager(
            compensation_pairs={"book": "cancel"},
            action_executor=executor,
        )

        record = manager.record_action("book", {"dest": "NYC"})
        manager.mark_completed(record.id, result={"id": "123"})

        result = manager.rollback()

        assert result.success is True
        assert record.id in result.compensated
        assert manager.log.get(record.id).status == ActionStatus.COMPENSATED

    def test_rollback_order(self):
        """Test rollback respects dependency order."""
        call_order = []

        class OrderTrackingExecutor:
            def execute(self, action, params):
                call_order.append(action)
                return {"status": "ok"}

        manager = RecoveryManager(
            compensation_pairs={"step1": "undo1", "step2": "undo2"},
            action_executor=OrderTrackingExecutor(),
            infer_dependencies=False,  # Disable inference, set manually
        )

        rec1 = manager.record_action("step1", {"key": "value1"})
        manager.mark_completed(rec1.id, result={"result_id": "UNIQUE123"})

        # Manually add dependency - rec2 depends on rec1
        from react_agent_compensation.core.models import ActionRecord
        rec2 = ActionRecord(
            action="step2",
            params={"input": "UNIQUE123"},
            compensator="undo2",
            depends_on=[rec1.id],  # Explicit dependency
        )
        manager.log.add(rec2)
        manager.mark_completed(rec2.id, result={"result_id": "R2"})

        manager.rollback()

        # Dependent (step2) should be undone before step1
        assert call_order.index("undo2") < call_order.index("undo1")

    def test_rollback_failure(self):
        """Test partial rollback failure."""
        executor = MockExecutor(
            results={"cancel1": {"status": "ok"}},
            errors={"cancel2": RuntimeError("compensation failed")},
        )
        manager = RecoveryManager(
            compensation_pairs={"action1": "cancel1", "action2": "cancel2"},
            action_executor=executor,
        )

        rec1 = manager.record_action("action1", {})
        manager.mark_completed(rec1.id, result={"id": "1"})

        rec2 = manager.record_action("action2", {})
        manager.mark_completed(rec2.id, result={"id": "2"})

        with pytest.raises(RollbackFailure) as exc_info:
            manager.rollback()

        assert len(exc_info.value.failed_records) > 0

    def test_skips_non_compensatable(self):
        """Test rollback skips actions without compensators."""
        executor = MockExecutor(
            results={"cancel": {"status": "ok"}},
        )
        manager = RecoveryManager(
            compensation_pairs={"book": "cancel"},  # Only book has compensator
            action_executor=executor,
        )

        rec1 = manager.record_action("book", {})
        manager.mark_completed(rec1.id, result={"id": "1"})

        rec2 = manager.record_action("read_only", {})  # No compensator
        manager.mark_completed(rec2.id, result={"data": "info"})

        result = manager.rollback()

        assert result.success is True
        assert rec1.id in result.compensated
        assert rec2.id not in result.compensated


class TestRecoveryResult:
    """Tests for RecoveryResult model."""

    def test_create_result(self):
        """Test creating a recovery result."""
        result = RecoveryResult(
            success=True,
            action_taken="retry",
            result={"id": "123"},
            attempts=2,
        )

        assert result.success is True
        assert result.action_taken == "retry"
        assert result.attempts == 2


class TestRollbackResult:
    """Tests for RollbackResult model."""

    def test_create_result(self):
        """Test creating a rollback result."""
        result = RollbackResult(
            success=True,
            compensated=["rec1", "rec2"],
            failed=[],
            message="Rolled back 2 actions",
        )

        assert result.success is True
        assert len(result.compensated) == 2
        assert result.message == "Rolled back 2 actions"

