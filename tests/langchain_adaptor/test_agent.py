"""Tests for LangChain agent factory."""

import pytest

from react_agent_compensation.core.transaction_log import TransactionLog
from react_agent_compensation.langchain_adaptor.agent import (
    create_multi_agent_log,
    get_compensation_middleware,
)


class TestAgentFactory:
    """Tests for agent factory functions."""

    def test_create_multi_agent_log(self):
        """Test creating a shared log."""
        log = create_multi_agent_log()

        assert isinstance(log, TransactionLog)
        assert len(log) == 0

    def test_get_compensation_middleware_none(self):
        """Test getting middleware from non-compensated object."""

        class FakeAgent:
            pass

        agent = FakeAgent()
        result = get_compensation_middleware(agent)

        assert result is None

    def test_get_compensation_middleware_exists(self):
        """Test getting middleware from compensated agent."""
        from react_agent_compensation.langchain_adaptor.middleware import CompensationMiddleware

        class MockTool:
            def __init__(self, name: str):
                self.name = name

        class FakeAgent:
            pass

        tools = [MockTool("book"), MockTool("cancel")]
        middleware = CompensationMiddleware(
            compensation_mapping={"book": "cancel"},
            tools=tools,
        )

        agent = FakeAgent()
        agent._compensation_middleware = middleware

        result = get_compensation_middleware(agent)

        assert result is middleware


class TestMultiAgentLog:
    """Tests for multi-agent log functionality."""

    def test_shared_log_isolation(self):
        """Test that multiple logs are independent."""
        log1 = create_multi_agent_log()
        log2 = create_multi_agent_log()

        assert log1 is not log2

    def test_shared_log_operations(self):
        """Test basic operations on shared log."""
        from react_agent_compensation.core.models import ActionRecord, ActionStatus

        log = create_multi_agent_log()

        # Add records from different agents
        rec1 = ActionRecord(action="action1", params={}, agent_id="agent1")
        rec2 = ActionRecord(action="action2", params={}, agent_id="agent2")

        log.add(rec1)
        log.add(rec2)

        assert len(log) == 2

        # Filter by agent
        agent1_records = log.filter_by_agent("agent1")
        assert len(agent1_records) == 1
        assert agent1_records[0].agent_id == "agent1"

