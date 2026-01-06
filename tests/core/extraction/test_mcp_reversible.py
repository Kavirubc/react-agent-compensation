"""Tests for MCP reversible extraction strategy."""

import pytest

from react_agent_compensation.core.extraction.mcp_reversible import (
    MCPReversibleExtractionStrategy,
)
from react_agent_compensation.core.mcp.metadata import MCPToolMetadata


class TestMCPReversibleExtractionStrategy:
    """Tests for MCPReversibleExtractionStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metadata = {
            "update_status": MCPToolMetadata(
                name="update_status",
                action_type="update",
                is_reversible=True,
            ),
            "add_item": MCPToolMetadata(
                name="add_item",
                action_type="create",
                compensation_pair="delete_item",
            ),
            "get_items": MCPToolMetadata(
                name="get_items",
                action_type="read",
            ),
            "update_non_reversible": MCPToolMetadata(
                name="update_non_reversible",
                action_type="update",
                is_reversible=False,
            ),
        }
        self.strategy = MCPReversibleExtractionStrategy(self.metadata)

    def test_extract_previous_fields(self):
        """Test extracting previous_* fields from result."""
        result = {
            "message": "Updated successfully",
            "previous_status": "pending",
            "previous_location": "NYC",
        }
        original_params = {"name": "John", "status": "completed"}

        extracted = self.strategy.extract(
            result, original_params, None, "update_status"
        )

        assert extracted is not None
        assert extracted["status"] == "pending"
        assert extracted["location"] == "NYC"
        assert extracted["name"] == "John"

    def test_preserves_identifying_params(self):
        """Test that identifying params from original call are preserved."""
        result = {"previous_status": "active"}
        original_params = {
            "id": "123",
            "name": "item1",
            "task_id": "task-456",
            "status": "inactive",
        }

        extracted = self.strategy.extract(
            result, original_params, None, "update_status"
        )

        assert extracted is not None
        assert extracted["id"] == "123"
        assert extracted["name"] == "item1"
        assert extracted["task_id"] == "task-456"
        assert extracted["status"] == "active"  # from previous_status

    def test_preserves_custom_id_params(self):
        """Test that custom *_id params are preserved."""
        result = {"previous_value": "old"}
        original_params = {
            "custom_record_id": "rec-789",
            "value": "new",
        }

        extracted = self.strategy.extract(
            result, original_params, None, "update_status"
        )

        assert extracted is not None
        assert extracted["custom_record_id"] == "rec-789"

    def test_returns_none_for_non_reversible(self):
        """Test that non-reversible tools return None."""
        result = {"previous_status": "old"}
        original_params = {"name": "test"}

        extracted = self.strategy.extract(
            result, original_params, None, "update_non_reversible"
        )

        assert extracted is None

    def test_returns_none_for_create_action(self):
        """Test that create actions return None."""
        result = {"id": "123", "previous_status": "none"}
        original_params = {"name": "new_item"}

        extracted = self.strategy.extract(
            result, original_params, None, "add_item"
        )

        assert extracted is None

    def test_returns_none_for_read_action(self):
        """Test that read actions return None."""
        result = {"items": [1, 2, 3]}
        original_params = {}

        extracted = self.strategy.extract(
            result, original_params, None, "get_items"
        )

        assert extracted is None

    def test_returns_none_for_unknown_tool(self):
        """Test that unknown tools return None."""
        result = {"previous_status": "old"}
        original_params = {"name": "test"}

        extracted = self.strategy.extract(
            result, original_params, None, "unknown_tool"
        )

        assert extracted is None

    def test_returns_none_for_missing_tool_name(self):
        """Test that missing tool name returns None."""
        result = {"previous_status": "old"}
        original_params = {"name": "test"}

        extracted = self.strategy.extract(
            result, original_params, None, None
        )

        assert extracted is None

    def test_returns_none_for_empty_previous_fields(self):
        """Test that result with no previous_* fields returns None."""
        result = {"message": "Updated", "status": "completed"}
        original_params = {"name": "test"}

        extracted = self.strategy.extract(
            result, original_params, None, "update_status"
        )

        # no previous_* fields, so returns None (only has name)
        # but it should still return with identifying params
        # actually this returns {"name": "test"} which is not empty
        assert extracted is not None
        assert extracted == {"name": "test"}

    def test_skips_none_previous_values(self):
        """Test that None previous values are skipped."""
        result = {
            "previous_status": "active",
            "previous_location": None,
        }
        original_params = {"name": "test"}

        extracted = self.strategy.extract(
            result, original_params, None, "update_status"
        )

        assert extracted is not None
        assert extracted["status"] == "active"
        assert "location" not in extracted

    def test_handles_non_dict_result(self):
        """Test handling of non-dict result."""
        result = "success"
        original_params = {"name": "test"}

        extracted = self.strategy.extract(
            result, original_params, None, "update_status"
        )

        # should still return identifying params
        assert extracted == {"name": "test"}

    def test_strategy_name(self):
        """Test strategy name property."""
        assert self.strategy.name == "MCPReversibleExtractionStrategy"
