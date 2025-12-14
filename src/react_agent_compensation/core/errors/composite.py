"""Composite strategy and factory for error detection chains.

This module provides:
- CompositeErrorStrategy: Chains multiple strategies with fallback
- create_error_detector: Factory function for common configurations
"""

from __future__ import annotations

from typing import Any

from react_agent_compensation.core.errors.base import ErrorStrategy
from react_agent_compensation.core.errors.content import ContentDictStrategy
from react_agent_compensation.core.errors.exception import ExceptionContentStrategy
from react_agent_compensation.core.errors.explicit import ExplicitStatusStrategy


class CompositeErrorStrategy(ErrorStrategy):
    """Chains multiple error detection strategies together.

    Strategies are evaluated in order until one returns a definitive
    answer (True or False). If all return None, uses the default.

    Example:
        strategy = CompositeErrorStrategy([
            ExplicitStatusStrategy(),
            ContentDictStrategy(),
            ExceptionContentStrategy(),
        ])

        is_err = strategy.is_error(result)
    """

    def __init__(
        self,
        strategies: list[ErrorStrategy] | None = None,
        default_is_error: bool = False,
    ):
        """Initialize composite strategy.

        Args:
            strategies: Ordered list of strategies. If None, uses defaults.
            default_is_error: Default return value if no strategy decides.
        """
        self.strategies = strategies or [
            ExplicitStatusStrategy(),
            ContentDictStrategy(),
            ExceptionContentStrategy(),
        ]
        self.default_is_error = default_is_error

    def is_error(self, result: Any) -> bool | None:
        """Check if result is an error using strategy chain.

        Args:
            result: The result to check

        Returns:
            True if error, False if not error, or default if undetermined
        """
        for strategy in self.strategies:
            decision = strategy.is_error(result)
            if decision is not None:
                return decision
        return self.default_is_error

    @property
    def name(self) -> str:
        """Return composite name with all strategy names."""
        strategy_names = [s.name for s in self.strategies]
        return f"CompositeErrorStrategy({', '.join(strategy_names)})"

    def get_error_message(self, result: Any) -> str | None:
        """Get error message from first strategy that can extract it.

        Args:
            result: The result to extract message from

        Returns:
            Error message if found by any strategy
        """
        for strategy in self.strategies:
            message = strategy.get_error_message(result)
            if message:
                return message
        return None

    def add_strategy(self, strategy: ErrorStrategy, priority: int | None = None) -> None:
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


class AlwaysErrorStrategy(ErrorStrategy):
    """Strategy that always returns True (for testing/debugging)."""

    def is_error(self, result: Any) -> bool | None:
        """Always return True."""
        return True


class NeverErrorStrategy(ErrorStrategy):
    """Strategy that always returns False (for testing/debugging)."""

    def is_error(self, result: Any) -> bool | None:
        """Always return False."""
        return False


def create_error_detector(
    strategies: list[ErrorStrategy] | None = None,
    include_explicit: bool = True,
    include_content: bool = True,
    include_exception: bool = True,
    default_is_error: bool = False,
) -> CompositeErrorStrategy:
    """Factory function to create a configured error detector.

    Creates a CompositeErrorStrategy with strategies in priority order:
    1. ExplicitStatusStrategy (if include_explicit)
    2. ContentDictStrategy (if include_content)
    3. ExceptionContentStrategy (if include_exception)

    Args:
        strategies: Custom strategies to use instead of defaults
        include_explicit: Include status attribute checking
        include_content: Include content dict checking
        include_exception: Include exception string checking
        default_is_error: Default when no strategy decides

    Returns:
        Configured CompositeErrorStrategy instance

    Example:
        # Default configuration
        detector = create_error_detector()

        # Minimal - only check explicit status
        detector = create_error_detector(
            include_content=False,
            include_exception=False,
        )

        # Custom strategies
        detector = create_error_detector(
            strategies=[MyCustomStrategy()],
        )
    """
    if strategies is not None:
        return CompositeErrorStrategy(
            strategies=strategies,
            default_is_error=default_is_error,
        )

    built_strategies: list[ErrorStrategy] = []

    if include_explicit:
        built_strategies.append(ExplicitStatusStrategy())

    if include_content:
        built_strategies.append(ContentDictStrategy())

    if include_exception:
        built_strategies.append(ExceptionContentStrategy())

    return CompositeErrorStrategy(
        strategies=built_strategies,
        default_is_error=default_is_error,
    )
