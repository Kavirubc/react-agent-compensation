"""Tests for TransactionLog."""

import pytest

from react_agent_compensation.core.models import ActionRecord, ActionStatus
from react_agent_compensation.core.transaction_log import TransactionLog


class TestTransactionLog:
    """Tests for TransactionLog."""

    def test_add_and_get(self):
        """Test adding and retrieving records."""
        log = TransactionLog()
        record = ActionRecord(action="test", params={})
        log.add(record)

        retrieved = log.get(record.id)
        assert retrieved is not None
        assert retrieved.action == "test"

    def test_get_nonexistent(self):
        """Test getting nonexistent record."""
        log = TransactionLog()
        assert log.get("nonexistent") is None

    def test_update(self):
        """Test updating a record."""
        log = TransactionLog()
        record = ActionRecord(action="test", params={})
        log.add(record)

        log.update(record.id, status=ActionStatus.COMPLETED, result={"id": "123"})

        updated = log.get(record.id)
        assert updated.status == ActionStatus.COMPLETED
        assert updated.result == {"id": "123"}

    def test_update_nonexistent(self):
        """Test updating nonexistent record raises error."""
        log = TransactionLog()
        with pytest.raises(KeyError):
            log.update("nonexistent", status=ActionStatus.COMPLETED)

    def test_snapshot(self):
        """Test snapshot creates deep copy."""
        log = TransactionLog()
        record = ActionRecord(action="test", params={"key": "value"})
        log.add(record)

        snapshot = log.snapshot()

        # Modify original
        log.update(record.id, status=ActionStatus.COMPLETED)

        # Snapshot should be unchanged
        assert snapshot[record.id].status == ActionStatus.PENDING

    def test_mark_compensated(self):
        """Test marking record as compensated."""
        log = TransactionLog()
        record = ActionRecord(action="test", params={}, compensator="undo")
        log.add(record)
        log.update(record.id, status=ActionStatus.COMPLETED)

        log.mark_compensated(record.id)

        updated = log.get(record.id)
        assert updated.status == ActionStatus.COMPENSATED
        assert updated.compensated is True

    def test_clear_all(self):
        """Test clearing all records."""
        log = TransactionLog()
        log.add(ActionRecord(action="test1", params={}))
        log.add(ActionRecord(action="test2", params={}))

        assert len(log) == 2
        log.clear()
        assert len(log) == 0

    def test_clear_by_agent(self):
        """Test clearing records by agent_id."""
        log = TransactionLog()
        log.add(ActionRecord(action="test1", params={}, agent_id="agent1"))
        log.add(ActionRecord(action="test2", params={}, agent_id="agent2"))

        log.clear(agent_id="agent1")

        assert len(log) == 1
        records = log.filter_by_agent("agent2")
        assert len(records) == 1

    def test_filter_by_status(self):
        """Test filtering by status."""
        log = TransactionLog()
        rec1 = ActionRecord(action="test1", params={})
        rec2 = ActionRecord(action="test2", params={})
        log.add(rec1)
        log.add(rec2)
        log.update(rec1.id, status=ActionStatus.COMPLETED)

        completed = log.filter_by_status(ActionStatus.COMPLETED)
        pending = log.filter_by_status(ActionStatus.PENDING)

        assert len(completed) == 1
        assert len(pending) == 1

    def test_contains(self):
        """Test __contains__ method."""
        log = TransactionLog()
        record = ActionRecord(action="test", params={})
        log.add(record)

        assert record.id in log
        assert "nonexistent" not in log


class TestRollbackPlan:
    """Tests for rollback plan generation."""

    def test_empty_log(self):
        """Test rollback plan with empty log."""
        log = TransactionLog()
        plan = log.get_rollback_plan()
        assert plan == []

    def test_no_compensatable(self):
        """Test rollback plan with no compensatable records."""
        log = TransactionLog()
        record = ActionRecord(action="test", params={})  # No compensator
        log.add(record)
        log.update(record.id, status=ActionStatus.COMPLETED)

        plan = log.get_rollback_plan()
        assert plan == []

    def test_simple_rollback(self):
        """Test simple rollback plan with one record."""
        log = TransactionLog()
        record = ActionRecord(action="test", params={}, compensator="undo_test")
        log.add(record)
        log.update(record.id, status=ActionStatus.COMPLETED)

        plan = log.get_rollback_plan()
        assert len(plan) == 1
        assert plan[0].id == record.id

    def test_dependency_ordering(self):
        """Test rollback plan respects dependencies."""
        log = TransactionLog()

        # Record 1 - no dependencies
        rec1 = ActionRecord(action="action1", params={}, compensator="undo1")
        log.add(rec1)
        log.update(rec1.id, status=ActionStatus.COMPLETED)

        # Record 2 - depends on record 1
        rec2 = ActionRecord(
            action="action2",
            params={},
            compensator="undo2",
            depends_on=[rec1.id],
        )
        log.add(rec2)
        log.update(rec2.id, status=ActionStatus.COMPLETED)

        plan = log.get_rollback_plan()

        # Record 2 should come before record 1 (dependents first)
        assert len(plan) == 2
        ids = [r.id for r in plan]
        assert ids.index(rec2.id) < ids.index(rec1.id)

    def test_excludes_pending(self):
        """Test rollback plan excludes pending records."""
        log = TransactionLog()
        rec1 = ActionRecord(action="action1", params={}, compensator="undo1")
        rec2 = ActionRecord(action="action2", params={}, compensator="undo2")
        log.add(rec1)
        log.add(rec2)
        log.update(rec1.id, status=ActionStatus.COMPLETED)
        # rec2 stays PENDING

        plan = log.get_rollback_plan()
        assert len(plan) == 1
        assert plan[0].id == rec1.id

    def test_excludes_already_compensated(self):
        """Test rollback plan excludes already compensated records."""
        log = TransactionLog()
        record = ActionRecord(action="test", params={}, compensator="undo")
        log.add(record)
        log.update(record.id, status=ActionStatus.COMPLETED)
        log.mark_compensated(record.id)

        plan = log.get_rollback_plan()
        assert plan == []

    def test_filter_by_agent(self):
        """Test rollback plan filtered by agent_id."""
        log = TransactionLog()

        rec1 = ActionRecord(action="a1", params={}, compensator="u1", agent_id="agent1")
        rec2 = ActionRecord(action="a2", params={}, compensator="u2", agent_id="agent2")
        log.add(rec1)
        log.add(rec2)
        log.update(rec1.id, status=ActionStatus.COMPLETED)
        log.update(rec2.id, status=ActionStatus.COMPLETED)

        plan = log.get_rollback_plan(agent_id="agent1")
        assert len(plan) == 1
        assert plan[0].agent_id == "agent1"


class TestSerialization:
    """Tests for log serialization."""

    def test_to_dict_and_from_dict(self):
        """Test round-trip serialization."""
        log = TransactionLog()
        rec1 = ActionRecord(action="test1", params={"a": 1}, compensator="undo1")
        rec2 = ActionRecord(action="test2", params={"b": 2})
        log.add(rec1)
        log.add(rec2)
        log.update(rec1.id, status=ActionStatus.COMPLETED, result={"id": "123"})

        # Serialize
        data = log.to_dict()
        assert "records" in data
        assert len(data["records"]) == 2

        # Deserialize
        restored = TransactionLog.from_dict(data)
        assert len(restored) == 2

        restored_rec1 = restored.get(rec1.id)
        assert restored_rec1.action == "test1"
        assert restored_rec1.status == ActionStatus.COMPLETED
        assert restored_rec1.result == {"id": "123"}
