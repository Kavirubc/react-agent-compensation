"""Composite strategy and factory for extraction strategy chains.

This module provides:
- CompositeExtractionStrategy: Chains multiple strategies with fallback
- create_extraction_strategy: Factory function for common configurations
"""

from __future__ import annotations

from typing import Any, Callable

from react_agent_compensation.core.extraction.base import (
    CompensationSchema,
    ExtractionStrategy,
    ToolLike,
)
from react_agent_compensation.core.extraction.heuristic import HeuristicExtractionStrategy
from react_agent_compensation.core.extraction.passthrough import PassthroughStrategy
from react_agent_compensation.core.extraction.recursive import RecursiveSearchStrategy
from react_agent_compensation.core.extraction.schema import SchemaExtractionStrategy
from react_agent_compensation.core.extraction.state_mappers import (
    StateMapperFn,
    StateMappersStrategy,
)


class CompositeExtractionStrategy(ExtractionStrategy):
    """Chains multiple extraction strategies together.

    Strategies are evaluated in order until one returns a non-None
    result. This implements the priority chain.

    Example:
        strategy = CompositeExtractionStrategy([
            StateMappersStrategy(state_mappers={...}),
            SchemaExtractionStrategy(schemas={...}),
            HeuristicExtractionStrategy(),
            RecursiveSearchStrategy(),
            PassthroughStrategy(),
        ])

        params = strategy.extract(result, original_params, comp_tool, tool_name)
    """

    def __init__(
        self,
        strategies: list[ExtractionStrategy] | None = None,
        raise_on_failure: bool = True,
    ):
        """Initialize composite strategy.

        Args:
            strategies: Ordered list of strategies. If None, uses defaults.
            raise_on_failure: If True, raises ValueError when no strategy
                succeeds. If False, returns None.
        """
        self.strategies = strategies or [
            HeuristicExtractionStrategy(),
            RecursiveSearchStrategy(),
            PassthroughStrategy(),
        ]
        self.raise_on_failure = raise_on_failure

    def extract(
        self,
        result: Any,
        original_params: dict[str, Any],
        compensation_tool: ToolLike | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Extract parameters using the strategy chain.

        Args:
            result: The result from the original tool call
            original_params: The original parameters passed to the tool
            compensation_tool: The compensation tool (for schema inspection)
            tool_name: Name of the original tool (for lookup)

        Returns:
            Extracted parameters from first successful strategy

        Raises:
            ValueError: If raise_on_failure=True and no strategy succeeds
        """
        for strategy in self.strategies:
            extracted = strategy.extract(
                result, original_params, compensation_tool, tool_name
            )
            if extracted is not None:
                return extracted

        if self.raise_on_failure:
            raise ValueError(
                f"No extraction strategy could extract parameters for tool "
                f"'{tool_name}' from result: {result}"
            )
        return None

    @property
    def name(self) -> str:
        """Return composite name with all strategy names."""
        strategy_names = [s.name for s in self.strategies]
        return f"CompositeExtractionStrategy({', '.join(strategy_names)})"

    def add_strategy(self, strategy: ExtractionStrategy, priority: int | None = None) -> None:
        """Add a strategy to the chain.

        Args:
            strategy: The strategy to add
            priority: Index to insert at (0 = highest priority).
                If None, appends to end.
        """
        if priority is not None:
            self.strategies.insert(priority, strategy)
        else:
            self.strategies.append(strategy)


def create_extraction_strategy(
    state_mappers: dict[str, StateMapperFn] | None = None,
    compensation_schemas: dict[str, CompensationSchema] | None = None,
    include_llm: bool = False,
    llm_model: str | None = None,
    raise_on_failure: bool = True,
) -> CompositeExtractionStrategy:
    """Factory function to create a configured extraction strategy chain.

    Creates a CompositeExtractionStrategy with strategies in priority order:
    1. StateMappersStrategy (if state_mappers provided)
    2. SchemaExtractionStrategy (if schemas provided)
    3. HeuristicExtractionStrategy (always)
    4. RecursiveSearchStrategy (always)
    5. LLMExtractionStrategy (if include_llm=True)
    6. PassthroughStrategy (always, last resort)

    Args:
        state_mappers: Custom mapping functions by tool name
        compensation_schemas: CompensationSchema instances by tool name
        include_llm: If True, adds LLM extraction before passthrough
        llm_model: Model to use for LLM extraction (e.g., "gpt-4o-mini")
        raise_on_failure: Whether to raise error if extraction fails

    Returns:
        Configured CompositeExtractionStrategy instance

    Example:
        extractor = create_extraction_strategy(
            state_mappers={
                "book_flight": lambda r, p: {"booking_id": r["id"]},
            },
            compensation_schemas={
                "book_hotel": CompensationSchema(
                    param_mapping={"reservation_id": "result.id"},
                ),
            },
            include_llm=True,
            llm_model="gpt-4o-mini",
        )
    """
    strategies: list[ExtractionStrategy] = []

    # Priority 1: Developer-provided state mappers
    if state_mappers:
        strategies.append(StateMappersStrategy(state_mappers))

    # Priority 2: Declarative schemas
    if compensation_schemas:
        strategies.append(SchemaExtractionStrategy(compensation_schemas))

    # Priority 3: Heuristic extraction (always included)
    strategies.append(HeuristicExtractionStrategy())

    # Priority 4: Recursive search
    strategies.append(RecursiveSearchStrategy())

    # Priority 5: LLM extraction (lazy-loaded from llm module)
    if include_llm:
        try:
            from react_agent_compensation.llm.extraction import LLMExtractionStrategy
            strategies.append(LLMExtractionStrategy(model=llm_model or "gpt-4o-mini"))
        except ImportError:
            pass  # LLM dependencies not installed

    # Priority 6: Passthrough (last resort)
    strategies.append(PassthroughStrategy())

    return CompositeExtractionStrategy(
        strategies=strategies,
        raise_on_failure=raise_on_failure,
    )
