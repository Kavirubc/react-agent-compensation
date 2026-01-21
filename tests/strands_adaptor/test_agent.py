"""Tests for Strands agent module."""

import pytest

from react_agent_compensation.strands_adaptor.agent import (
    get_compensation_provider,
    wrap_tools_with_compensation,
)
from react_agent_compensation.strands_adaptor.hooks import CompensationHookProvider


class TestWrapToolsWithCompensation:
    """Tests for wrap_tools_with_compensation function."""

    def test_returns_provider_and_tools(self, tools_list, compensation_mapping):
        """Test returns tuple of provider and tools."""
        provider, tools = wrap_tools_with_compensation(
            tools=tools_list,
            compensation_mapping=compensation_mapping,
        )

        assert isinstance(provider, CompensationHookProvider)
        assert isinstance(tools, list)
        assert len(tools) == len(tools_list)

    def test_provider_has_correct_mapping(self, tools_list, compensation_mapping):
        """Test provider has correct compensation mapping."""
        provider, _ = wrap_tools_with_compensation(
            tools=tools_list,
            compensation_mapping=compensation_mapping,
        )

        assert provider.rc_manager.is_compensatable("reserve_inventory")
        assert provider.rc_manager.is_compensatable("process_payment")
        assert not provider.rc_manager.is_compensatable("release_inventory")

    def test_with_goals(self, tools_list, compensation_mapping):
        """Test provider includes goals."""
        provider, _ = wrap_tools_with_compensation(
            tools=tools_list,
            compensation_mapping=compensation_mapping,
            goals=["fast_processing"],
        )

        assert provider._goals == ["fast_processing"]


class TestGetCompensationProvider:
    """Tests for get_compensation_provider function."""

    def test_returns_none_for_plain_object(self):
        """Test returns None for objects without provider."""

        class PlainAgent:
            pass

        result = get_compensation_provider(PlainAgent())

        assert result is None

    def test_returns_provider_if_present(self, tools_list, compensation_mapping):
        """Test returns provider if _compensation_provider is set."""

        class MockAgent:
            pass

        agent = MockAgent()
        provider = CompensationHookProvider(
            compensation_pairs=compensation_mapping,
            tools=tools_list,
        )
        agent._compensation_provider = provider

        result = get_compensation_provider(agent)

        assert result is provider


# Note: Tests for create_compensated_agent require Strands to be installed.


class TestAgentIntegration:
    """Integration tests requiring Strands."""

    @pytest.mark.skip(reason="Requires Strands installation")
    def test_create_compensated_agent(self):
        """Test creating a compensated agent."""
        from react_agent_compensation.strands_adaptor import create_compensated_agent

        # Would test agent creation here
        pass

    @pytest.mark.skip(reason="Requires Strands installation")
    def test_agent_async(self):
        """Test async agent execution."""
        from react_agent_compensation.strands_adaptor import (
            create_compensated_agent_async,
        )

        # Would test async agent here
        pass
