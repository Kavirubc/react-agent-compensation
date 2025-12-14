"""Tests for core data models."""

import time

import pytest

from react_agent_compensation.core.models import ActionRecord, ActionStatus


class TestActionStatus:
    """Tests for ActionStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert ActionStatus.PENDING == "PENDING"
        assert ActionStatus.COMPLETED == "COMPLETED"
        assert ActionStatus.FAILED == "FAILED"
        assert ActionStatus.COMPENSATED == "COMPENSATED"


class TestActionRecord:
    """Tests for ActionRecord model."""

    def test_create_record(self):
        """Test creating a basic action record."""
        record = ActionRecord(action="test_tool", params={"key": "value"})

        assert record.action == "test_tool"
        assert record.params == {"key": "value"}
        assert record.status == ActionStatus.PENDING
        assert record.result is None
        assert record.compensator is None
        assert record.depends_on == []
        assert record.compensated is False
        assert record.id is not None
        assert record.timestamp > 0

    def test_create_with_compensator(self):
        """Test creating record with compensator."""
        record = ActionRecord(
            action="book_flight",
            params={"dest": "NYC"},
            compensator="cancel_flight",
        )

        assert record.compensator == "cancel_flight"

    def test_create_with_dependencies(self):
        """Test creating record with dependencies."""
        record = ActionRecord(
            action="book_hotel",
            params={},
            depends_on=["rec-1", "rec-2"],
        )

        assert record.depends_on == ["rec-1", "rec-2"]

    def test_mark_completed(self):
        """Test marking record as completed."""
        record = ActionRecord(action="test", params={})
        record.mark_completed(result={"id": "123"})

        assert record.status == ActionStatus.COMPLETED
        assert record.result == {"id": "123"}

    def test_mark_failed(self):
        """Test marking record as failed."""
        record = ActionRecord(action="test", params={})
        record.mark_failed(error="Connection error")

        assert record.status == ActionStatus.FAILED
        assert record.result == {"error": "Connection error"}

    def test_mark_compensated(self):
        """Test marking record as compensated."""
        record = ActionRecord(action="test", params={})
        record.mark_completed(result={})
        record.mark_compensated()

        assert record.status == ActionStatus.COMPENSATED
        assert record.compensated is True

    def test_is_compensatable(self):
        """Test is_compensatable check."""
        # Not compensatable - no compensator
        record1 = ActionRecord(action="test", params={})
        record1.mark_completed(result={})
        assert record1.is_compensatable() is False

        # Not compensatable - pending status
        record2 = ActionRecord(action="test", params={}, compensator="undo_test")
        assert record2.is_compensatable() is False

        # Compensatable
        record3 = ActionRecord(action="test", params={}, compensator="undo_test")
        record3.mark_completed(result={})
        assert record3.is_compensatable() is True

        # Not compensatable - already compensated
        record3.mark_compensated()
        assert record3.is_compensatable() is False

    def test_serialization(self):
        """Test model serialization."""
        record = ActionRecord(
            action="test",
            params={"key": "value"},
            compensator="undo_test",
        )

        data = record.model_dump()
        assert data["action"] == "test"
        assert data["params"] == {"key": "value"}
        assert data["compensator"] == "undo_test"

        # Deserialize
        restored = ActionRecord.model_validate(data)
        assert restored.action == record.action
        assert restored.id == record.id
