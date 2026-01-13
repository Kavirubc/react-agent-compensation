"""Tests for MCP tool metadata parsing and helpers."""

import pytest

from react_agent_compensation.core.mcp.metadata import (
    MCPToolMetadata,
    get_compensator,
    is_compensatable,
    parse_tool_metadata,
    should_track_action,
)


class TestParseToolMetadata:
    """Tests for parse_tool_metadata function."""

    def test_parse_with_all_annotations(self):
        """Test parsing schema with all annotation fields."""
        schema = {
            "name": "update_status",
            "annotations": {
                "x-action-type": "update",
                "x-reversible": True,
                "x-destructive": False,
                "x-requires-confirmation": False,
                "x-category": "status",
                "x-compensation-pair": "revert_status",
            },
        }

        metadata = parse_tool_metadata(schema)

        assert metadata is not None
        assert metadata.name == "update_status"
        assert metadata.action_type == "update"
        assert metadata.is_reversible is True
        assert metadata.is_destructive is False
        assert metadata.requires_confirmation is False
        assert metadata.category == "status"
        assert metadata.compensation_pair == "revert_status"

    def test_parse_with_input_schema_annotations(self):
        """Test parsing annotations from inputSchema."""
        schema = {
            "name": "add_item",
            "inputSchema": {
                "type": "object",
                "x-action-type": "create",
                "x-compensation-pair": "delete_item",
            },
        }

        metadata = parse_tool_metadata(schema)

        assert metadata is not None
        assert metadata.action_type == "create"
        assert metadata.compensation_pair == "delete_item"

    def test_parse_minimal_schema(self):
        """Test parsing schema with only name."""
        schema = {"name": "simple_tool"}

        metadata = parse_tool_metadata(schema)

        assert metadata is not None
        assert metadata.name == "simple_tool"
        assert metadata.action_type is None
        assert metadata.compensation_pair is None
        assert metadata.is_reversible is False

    def test_parse_missing_name_returns_none(self):
        """Test that missing name returns None."""
        schema = {"description": "A tool without name"}

        metadata = parse_tool_metadata(schema)

        assert metadata is None

    def test_parse_annotations_priority(self):
        """Test that annotations dict takes priority over inputSchema."""
        schema = {
            "name": "tool",
            "inputSchema": {"x-action-type": "create"},
            "annotations": {"x-action-type": "update"},
        }

        metadata = parse_tool_metadata(schema)

        # annotations should override inputSchema
        assert metadata.action_type == "update"

    def test_parse_bool_from_string(self):
        """Test boolean parsing from string values."""
        schema = {
            "name": "tool",
            "annotations": {
                "x-reversible": "true",
                "x-destructive": "false",
            },
        }

        metadata = parse_tool_metadata(schema)

        assert metadata.is_reversible is True
        assert metadata.is_destructive is False


class TestShouldTrackAction:
    """Tests for should_track_action function."""

    def test_read_returns_false(self):
        """Test that read operations should not be tracked."""
        metadata = MCPToolMetadata(name="get_items", action_type="read")

        assert should_track_action(metadata) is False

    def test_create_returns_true(self):
        """Test that create operations should be tracked."""
        metadata = MCPToolMetadata(name="add_item", action_type="create")

        assert should_track_action(metadata) is True

    def test_update_returns_true(self):
        """Test that update operations should be tracked."""
        metadata = MCPToolMetadata(name="update_item", action_type="update")

        assert should_track_action(metadata) is True

    def test_delete_returns_true(self):
        """Test that delete operations should be tracked."""
        metadata = MCPToolMetadata(name="delete_item", action_type="delete")

        assert should_track_action(metadata) is True

    def test_none_action_type_returns_true(self):
        """Test that missing action type defaults to tracking."""
        metadata = MCPToolMetadata(name="unknown_tool", action_type=None)

        assert should_track_action(metadata) is True


class TestGetCompensator:
    """Tests for get_compensator function."""

    def test_reversible_update_returns_self(self):
        """Test that reversible updates return the tool's own name."""
        metadata = MCPToolMetadata(
            name="update_status",
            action_type="update",
            is_reversible=True,
        )

        compensator = get_compensator(metadata)

        assert compensator == "update_status"

    def test_create_with_pair_returns_pair(self):
        """Test that create with compensation pair returns the pair."""
        metadata = MCPToolMetadata(
            name="add_item",
            action_type="create",
            compensation_pair="delete_item",
        )

        compensator = get_compensator(metadata)

        assert compensator == "delete_item"

    def test_reversible_non_update_returns_pair(self):
        """Test that reversible flag on non-update uses pair."""
        metadata = MCPToolMetadata(
            name="add_item",
            action_type="create",
            is_reversible=True,
            compensation_pair="delete_item",
        )

        compensator = get_compensator(metadata)

        # only updates use self-compensation
        assert compensator == "delete_item"

    def test_no_compensation_returns_none(self):
        """Test that tool without compensation returns None."""
        metadata = MCPToolMetadata(name="read_only", action_type="read")

        compensator = get_compensator(metadata)

        assert compensator is None


class TestIsCompensatable:
    """Tests for is_compensatable function."""

    def test_with_pair_is_compensatable(self):
        """Test tool with compensation pair is compensatable."""
        metadata = MCPToolMetadata(
            name="add_item",
            compensation_pair="delete_item",
        )

        assert is_compensatable(metadata) is True

    def test_reversible_update_is_compensatable(self):
        """Test reversible update is compensatable."""
        metadata = MCPToolMetadata(
            name="update_status",
            action_type="update",
            is_reversible=True,
        )

        assert is_compensatable(metadata) is True

    def test_read_only_not_compensatable(self):
        """Test read-only tool is not compensatable."""
        metadata = MCPToolMetadata(name="get_items", action_type="read")

        assert is_compensatable(metadata) is False
