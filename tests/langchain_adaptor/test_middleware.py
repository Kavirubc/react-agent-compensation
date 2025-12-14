"""Tests for LangChain adaptor middleware."""

import pytest

from react_agent_compensation.core.config import RetryPolicy
from react_agent_compensation.core.extraction import CompensationSchema
from react_agent_compensation.core.models import ActionStatus
from react_agent_compensation.core.transaction_log import TransactionLog
from react_agent_compensation.langchain_adaptor.middleware import CompensationMiddleware


class MockTool:
    """Mock LangChain tool for testing."""

    def __init__(self, name: str, result: dict | None = None, error: Exception | None = None):
        self.name = name
        self._result = result or {"status": "ok"}
        self._error = error

    def invoke(self, params: dict) -> dict:
        if self._error:
            raise self._error
        return self._result


class TestCompensationMiddleware:
    """Tests for CompensationMiddleware."""

    def test_create_middleware(self):
        """Test creating middleware."""
        tools = [MockTool("book_flight"), MockTool("cancel_flight")]
        middleware = CompensationMiddleware(
            compensation_mapping={"book_flight": "cancel_flight"},
            tools=tools,
        )

        assert middleware.compensation_mapping == {"book_flight": "cancel_flight"}
        assert middleware.rc_manager is not None
        assert middleware.transaction_log is not None

    def test_create_with_alternatives(self):
        """Test creating with alternative map."""
        tools = [
            MockTool("book_primary"),
            MockTool("book_backup"),
            MockTool("cancel"),
        ]
        middleware = CompensationMiddleware(
            compensation_mapping={"book_primary": "cancel"},
            alternative_map={"book_primary": ["book_backup"]},
            tools=tools,
        )

        assert middleware.alternative_map == {"book_primary": ["book_backup"]}

    def test_create_with_retry_policy(self):
        """Test creating with retry policy."""
        tools = [MockTool("book"), MockTool("cancel")]
        policy = RetryPolicy(max_retries=5, initial_delay=0.5)

        middleware = CompensationMiddleware(
            compensation_mapping={"book": "cancel"},
            tools=tools,
            retry_policy=policy,
        )

        assert middleware.rc_manager._retry_policy.max_retries == 5

    def test_shared_log(self):
        """Test using shared transaction log."""
        shared_log = TransactionLog()
        tools = [MockTool("book"), MockTool("cancel")]

        middleware = CompensationMiddleware(
            compensation_mapping={"book": "cancel"},
            tools=tools,
            shared_log=shared_log,
        )

        assert middleware.transaction_log is shared_log

    def test_agent_id(self):
        """Test setting agent ID."""
        tools = [MockTool("book"), MockTool("cancel")]

        middleware = CompensationMiddleware(
            compensation_mapping={"book": "cancel"},
            tools=tools,
            agent_id="agent1",
        )

        assert middleware.rc_manager._agent_id == "agent1"

    def test_add_tool(self):
        """Test adding a tool dynamically."""
        tools = [MockTool("book")]
        middleware = CompensationMiddleware(
            compensation_mapping={"book": "cancel"},
            tools=tools,
        )

        new_tool = MockTool("cancel")
        middleware.add_tool(new_tool)

        assert "cancel" in middleware._tools_cache

    def test_clear(self):
        """Test clearing the log."""
        tools = [MockTool("book"), MockTool("cancel")]
        middleware = CompensationMiddleware(
            compensation_mapping={"book": "cancel"},
            tools=tools,
        )

        # Add some records manually
        middleware.rc_manager.record_action("book", {"dest": "NYC"})
        assert len(middleware.transaction_log) > 0

        middleware.clear()
        assert len(middleware.transaction_log) == 0

    def test_with_compensation_schemas(self):
        """Test with declarative extraction schemas."""
        tools = [MockTool("book"), MockTool("cancel")]
        schemas = {
            "book": CompensationSchema(
                param_mapping={"booking_id": "result.id"},
            ),
        }

        middleware = CompensationMiddleware(
            compensation_mapping={"book": "cancel"},
            tools=tools,
            compensation_schemas=schemas,
        )

        # Should use schema extraction
        assert middleware.rc_manager._extraction_strategy is not None

    def test_with_state_mappers(self):
        """Test with custom state mapper functions."""
        tools = [MockTool("book"), MockTool("cancel")]
        mappers = {
            "book": lambda r, p: {"booking_id": r["id"]},
        }

        middleware = CompensationMiddleware(
            compensation_mapping={"book": "cancel"},
            tools=tools,
            state_mappers=mappers,
        )

        assert middleware.rc_manager._extraction_strategy is not None


class TestMultiAgentScenario:
    """Tests for multi-agent scenarios."""

    def test_shared_log_across_middlewares(self):
        """Test that multiple middlewares share the same log."""
        shared_log = TransactionLog()

        tools1 = [MockTool("book_flight"), MockTool("cancel_flight")]
        tools2 = [MockTool("book_hotel"), MockTool("cancel_hotel")]

        middleware1 = CompensationMiddleware(
            compensation_mapping={"book_flight": "cancel_flight"},
            tools=tools1,
            shared_log=shared_log,
            agent_id="agent1",
        )

        middleware2 = CompensationMiddleware(
            compensation_mapping={"book_hotel": "cancel_hotel"},
            tools=tools2,
            shared_log=shared_log,
            agent_id="agent2",
        )

        # Both should share the same log
        assert middleware1.transaction_log is middleware2.transaction_log

        # Records from both should appear
        middleware1.rc_manager.record_action("book_flight", {"dest": "NYC"})
        middleware2.rc_manager.record_action("book_hotel", {"city": "NYC"})

        assert len(shared_log) == 2

    def test_filtered_rollback_by_agent(self):
        """Test rollback can be filtered by agent."""
        shared_log = TransactionLog()

        tools1 = [MockTool("book_flight"), MockTool("cancel_flight")]
        tools2 = [MockTool("book_hotel"), MockTool("cancel_hotel")]

        middleware1 = CompensationMiddleware(
            compensation_mapping={"book_flight": "cancel_flight"},
            tools=tools1,
            shared_log=shared_log,
            agent_id="agent1",
        )

        middleware2 = CompensationMiddleware(
            compensation_mapping={"book_hotel": "cancel_hotel"},
            tools=tools2,
            shared_log=shared_log,
            agent_id="agent2",
        )

        # Record and complete actions
        rec1 = middleware1.rc_manager.record_action("book_flight", {"dest": "NYC"})
        middleware1.rc_manager.mark_completed(rec1.id, {"id": "F123"})

        rec2 = middleware2.rc_manager.record_action("book_hotel", {"city": "NYC"})
        middleware2.rc_manager.mark_completed(rec2.id, {"id": "H456"})

        # Get rollback plan for agent1 only
        plan = shared_log.get_rollback_plan(agent_id="agent1")

        assert len(plan) == 1
        assert plan[0].agent_id == "agent1"


class TestIntercept:
    """Tests for tool call interception."""

    def test_records_compensatable_action(self):
        """Test that compensatable actions are recorded."""
        book_tool = MockTool("book", result={"id": "123"})
        cancel_tool = MockTool("cancel")

        middleware = CompensationMiddleware(
            compensation_mapping={"book": "cancel"},
            tools=[book_tool, cancel_tool],
        )

        # Simulate interception
        record = middleware.rc_manager.record_action("book", {"dest": "NYC"})
        result = book_tool.invoke({"dest": "NYC"})
        middleware.rc_manager.mark_completed(record.id, result)

        # Verify recorded
        stored = middleware.transaction_log.get(record.id)
        assert stored is not None
        assert stored.status == ActionStatus.COMPLETED
        assert stored.result == {"id": "123"}

    def test_non_compensatable_not_recorded(self):
        """Test that non-compensatable actions aren't tracked for rollback."""
        read_tool = MockTool("read_data", result={"data": [1, 2, 3]})

        middleware = CompensationMiddleware(
            compensation_mapping={},  # No compensation pairs
            tools=[read_tool],
        )

        # This action doesn't have a compensator
        record = middleware.rc_manager.record_action("read_data", {})
        middleware.rc_manager.mark_completed(record.id, {"data": [1, 2, 3]})

        # Record exists but no compensator
        stored = middleware.transaction_log.get(record.id)
        assert stored.compensator is None

