"""Tests for CrewAI factory module."""

import pytest

from react_agent_compensation.crewai_adaptor.factory import (
    create_shared_log,
    get_compensation_middleware,
)


class TestCreateSharedLog:
    """Tests for create_shared_log function."""

    def test_creates_transaction_log(self):
        """Test creates a new TransactionLog."""
        from react_agent_compensation.core.transaction_log import TransactionLog

        log = create_shared_log()

        assert isinstance(log, TransactionLog)

    def test_creates_unique_instances(self):
        """Test creates unique instances each time."""
        log1 = create_shared_log()
        log2 = create_shared_log()

        assert log1 is not log2


class TestGetCompensationMiddleware:
    """Tests for get_compensation_middleware function."""

    def test_returns_none_for_plain_object(self):
        """Test returns None for objects without middleware."""

        class PlainAgent:
            pass

        result = get_compensation_middleware(PlainAgent())

        assert result is None

    def test_returns_middleware_if_present(self):
        """Test returns middleware if _compensation_middleware is set."""
        from react_agent_compensation.crewai_adaptor.middleware import (
            CrewAICompensationMiddleware,
        )

        class MockAgent:
            pass

        agent = MockAgent()
        middleware = CrewAICompensationMiddleware(
            compensation_mapping={"Book": "Cancel"},
            tools=[],
        )
        agent._compensation_middleware = middleware

        result = get_compensation_middleware(agent)

        assert result is middleware


# Note: Tests for create_compensated_crew and create_compensated_agent
# require CrewAI to be installed. These are integration tests.


class TestFactoryIntegration:
    """Integration tests requiring CrewAI."""

    @pytest.mark.skip(reason="Requires CrewAI installation")
    def test_create_compensated_crew(self):
        """Test creating a compensated crew."""
        # Would test crew creation here with actual CrewAI SDK

    @pytest.mark.skip(reason="Requires CrewAI installation")
    def test_create_compensated_agent(self):
        """Test creating a compensated agent."""
        # Would test agent creation here with actual CrewAI SDK
