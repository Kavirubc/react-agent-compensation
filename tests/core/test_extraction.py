"""Tests for extraction strategies."""

import pytest

from react_agent_compensation.core.extraction import (
    CompensationSchema,
    CompositeExtractionStrategy,
    HeuristicExtractionStrategy,
    PassthroughStrategy,
    RecursiveSearchStrategy,
    SchemaExtractionStrategy,
    StateMappersStrategy,
    create_extraction_strategy,
    resolve_path,
    validate_path,
)


class TestPathResolver:
    """Tests for path resolution."""

    def test_simple_path(self):
        """Test simple dot notation."""
        context = {"result": {"id": "123"}}
        assert resolve_path("result.id", context) == "123"

    def test_nested_path(self):
        """Test nested dot notation."""
        context = {"result": {"data": {"booking": {"id": "ABC"}}}}
        assert resolve_path("result.data.booking.id", context) == "ABC"

    def test_array_index(self):
        """Test array index access."""
        context = {"result": {"items": ["a", "b", "c"]}}
        assert resolve_path("result.items[0]", context) == "a"
        assert resolve_path("result.items[2]", context) == "c"

    def test_mixed_access(self):
        """Test mixed dot and array access."""
        context = {"result": {"items": [{"id": "1"}, {"id": "2"}]}}
        assert resolve_path("result.items[0].id", context) == "1"
        assert resolve_path("result.items[1].id", context) == "2"

    def test_params_access(self):
        """Test accessing original params."""
        context = {
            "result": {"id": "123"},
            "params": {"origin": "NYC", "dest": "LAX"},
        }
        assert resolve_path("params.origin", context) == "NYC"

    def test_invalid_path(self):
        """Test invalid path raises error."""
        context = {"result": {"id": "123"}}
        with pytest.raises(KeyError):
            resolve_path("result.nonexistent", context)

    def test_validate_path(self):
        """Test path validation."""
        assert validate_path("result.id") is True
        assert validate_path("result.items[0].name") is True
        assert validate_path("result.field?") is True
        assert validate_path("") is False


class TestCompensationSchema:
    """Tests for CompensationSchema."""

    def test_simple_extraction(self):
        """Test simple field extraction."""
        schema = CompensationSchema(
            param_mapping={"booking_id": "result.id"},
        )
        result = {"id": "ABC123"}
        extracted = schema.extract(result, {})

        assert extracted == {"booking_id": "ABC123"}

    def test_with_static_params(self):
        """Test extraction with static params."""
        schema = CompensationSchema(
            param_mapping={"booking_id": "result.id"},
            static_params={"reason": "Auto rollback"},
        )
        result = {"id": "ABC123"}
        extracted = schema.extract(result, {})

        assert extracted == {"booking_id": "ABC123", "reason": "Auto rollback"}

    def test_from_params(self):
        """Test extracting from original params."""
        schema = CompensationSchema(
            param_mapping={
                "booking_id": "result.id",
                "origin": "params.origin",
            },
        )
        result = {"id": "ABC123"}
        original_params = {"origin": "NYC"}
        extracted = schema.extract(result, original_params)

        assert extracted == {"booking_id": "ABC123", "origin": "NYC"}

    def test_optional_field(self):
        """Test optional field with ?."""
        schema = CompensationSchema(
            param_mapping={
                "booking_id": "result.id",
                "note": "result.note?",  # Optional
            },
        )
        result = {"id": "ABC123"}
        extracted = schema.extract(result, {})

        assert extracted == {"booking_id": "ABC123"}

    def test_required_field_missing(self):
        """Test required field missing raises error."""
        schema = CompensationSchema(
            param_mapping={"booking_id": "result.id"},
        )
        result = {"status": "ok"}  # No "id" field

        with pytest.raises(ValueError, match="Cannot extract"):
            schema.extract(result, {})


class TestStateMappersStrategy:
    """Tests for StateMappersStrategy."""

    def test_mapper_found(self):
        """Test extraction when mapper exists."""
        mappers = {
            "book_flight": lambda r, p: {"booking_id": r["id"]},
        }
        strategy = StateMappersStrategy(mappers)

        result = strategy.extract(
            result={"id": "ABC123"},
            original_params={},
            tool_name="book_flight",
        )

        assert result == {"booking_id": "ABC123"}

    def test_mapper_not_found(self):
        """Test returns None when no mapper."""
        strategy = StateMappersStrategy({})

        result = strategy.extract(
            result={"id": "123"},
            original_params={},
            tool_name="unknown_tool",
        )

        assert result is None

    def test_mapper_error_returns_none(self):
        """Test mapper exception returns None."""
        mappers = {
            "bad_tool": lambda r, p: r["nonexistent"],  # Will raise KeyError
        }
        strategy = StateMappersStrategy(mappers)

        result = strategy.extract(
            result={"id": "123"},
            original_params={},
            tool_name="bad_tool",
        )

        assert result is None


class TestSchemaExtractionStrategy:
    """Tests for SchemaExtractionStrategy."""

    def test_schema_found(self):
        """Test extraction when schema exists."""
        schemas = {
            "book_flight": CompensationSchema(
                param_mapping={"booking_id": "result.id"},
            ),
        }
        strategy = SchemaExtractionStrategy(schemas)

        result = strategy.extract(
            result={"id": "ABC123"},
            original_params={},
            tool_name="book_flight",
        )

        assert result == {"booking_id": "ABC123"}

    def test_schema_not_found(self):
        """Test returns None when no schema."""
        strategy = SchemaExtractionStrategy({})

        result = strategy.extract(
            result={"id": "123"},
            original_params={},
            tool_name="unknown_tool",
        )

        assert result is None


class TestHeuristicExtractionStrategy:
    """Tests for HeuristicExtractionStrategy."""

    def test_finds_id_field(self):
        """Test finding common id field."""
        strategy = HeuristicExtractionStrategy()

        result = strategy.extract(
            result={"id": "123", "status": "ok"},
            original_params={},
        )

        assert result == {"id": "123"}

    def test_finds_booking_id(self):
        """Test finding booking_id field."""
        strategy = HeuristicExtractionStrategy()

        result = strategy.extract(
            result={"booking_id": "ABC", "amount": 100},
            original_params={},
        )

        assert result == {"booking_id": "ABC"}

    def test_string_result(self):
        """Test handling string result."""
        strategy = HeuristicExtractionStrategy()

        result = strategy.extract(
            result="ABC123",
            original_params={},
        )

        assert result == {"id": "ABC123"}

    def test_no_id_field(self):
        """Test returns None when no ID found."""
        strategy = HeuristicExtractionStrategy()

        result = strategy.extract(
            result={"status": "ok", "message": "done"},
            original_params={},
        )

        assert result is None

    def test_custom_id_fields(self):
        """Test custom ID field names."""
        strategy = HeuristicExtractionStrategy(id_fields=["custom_id"])

        result = strategy.extract(
            result={"custom_id": "123", "id": "456"},
            original_params={},
        )

        assert result == {"custom_id": "123"}


class TestRecursiveSearchStrategy:
    """Tests for RecursiveSearchStrategy."""

    def test_finds_nested_id(self):
        """Test finding deeply nested ID."""
        strategy = RecursiveSearchStrategy()

        result = strategy.extract(
            result={"data": {"booking": {"id": "ABC123"}}},
            original_params={},
        )

        assert result == {"id": "ABC123"}

    def test_max_depth(self):
        """Test respects max depth."""
        strategy = RecursiveSearchStrategy(max_depth=1)

        result = strategy.extract(
            result={"level1": {"level2": {"level3": {"id": "123"}}}},
            original_params={},
        )

        assert result is None  # Too deep

    def test_no_id_found(self):
        """Test returns None when no ID."""
        strategy = RecursiveSearchStrategy()

        result = strategy.extract(
            result={"data": {"status": "ok"}},
            original_params={},
        )

        assert result is None


class TestPassthroughStrategy:
    """Tests for PassthroughStrategy."""

    def test_passes_dict(self):
        """Test passes through dict result."""
        strategy = PassthroughStrategy()

        result = strategy.extract(
            result={"id": "123", "status": "ok"},
            original_params={},
        )

        assert result == {"id": "123", "status": "ok"}

    def test_rejects_non_dict(self):
        """Test returns None for non-dict."""
        strategy = PassthroughStrategy()

        result = strategy.extract(
            result="string value",
            original_params={},
        )

        assert result is None

    def test_allow_non_dict(self):
        """Test wrapping non-dict when allowed."""
        strategy = PassthroughStrategy(allow_non_dict=True)

        result = strategy.extract(
            result="string value",
            original_params={},
        )

        assert result == {"value": "string value"}


class TestCompositeExtractionStrategy:
    """Tests for CompositeExtractionStrategy."""

    def test_uses_first_successful(self):
        """Test uses first strategy that succeeds."""
        strategy = CompositeExtractionStrategy([
            HeuristicExtractionStrategy(),
            PassthroughStrategy(),
        ])

        result = strategy.extract(
            result={"id": "123", "extra": "data"},
            original_params={},
        )

        # Heuristic should find id first
        assert result == {"id": "123"}

    def test_falls_through(self):
        """Test falls through to next strategy."""
        strategy = CompositeExtractionStrategy([
            HeuristicExtractionStrategy(),
            PassthroughStrategy(),
        ])

        result = strategy.extract(
            result={"status": "ok", "message": "done"},  # No ID
            original_params={},
        )

        # Should fall through to passthrough
        assert result == {"status": "ok", "message": "done"}

    def test_raise_on_failure(self):
        """Test raises when all fail."""
        strategy = CompositeExtractionStrategy(
            [HeuristicExtractionStrategy()],
            raise_on_failure=True,
        )

        with pytest.raises(ValueError):
            strategy.extract(
                result={"no": "id"},
                original_params={},
                tool_name="test_tool",
            )

    def test_no_raise_returns_none(self):
        """Test returns None when raise_on_failure=False."""
        strategy = CompositeExtractionStrategy(
            [HeuristicExtractionStrategy()],
            raise_on_failure=False,
        )

        result = strategy.extract(
            result={"no": "id"},
            original_params={},
        )

        assert result is None


class TestCreateExtractionStrategy:
    """Tests for factory function."""

    def test_default_strategies(self):
        """Test default strategy chain."""
        strategy = create_extraction_strategy()

        # Should include heuristic, recursive, passthrough
        assert isinstance(strategy, CompositeExtractionStrategy)
        assert len(strategy.strategies) >= 3

    def test_with_mappers(self):
        """Test with custom mappers."""
        mappers = {"test": lambda r, p: {"id": r["id"]}}
        strategy = create_extraction_strategy(state_mappers=mappers)

        # Mappers should be first
        assert isinstance(strategy.strategies[0], StateMappersStrategy)

    def test_with_schemas(self):
        """Test with schemas."""
        schemas = {"test": CompensationSchema(param_mapping={})}
        strategy = create_extraction_strategy(compensation_schemas=schemas)

        # Should include schema strategy
        has_schema = any(
            isinstance(s, SchemaExtractionStrategy) for s in strategy.strategies
        )
        assert has_schema
