"""No-retry strategy - always fails immediately.

Use this when you want to disable retries for specific tools
or in testing scenarios.
"""

from __future__ import annotations

from react_agent_compensation.core.retry.base import RetryContext, RetryStrategy


class NoRetryStrategy(RetryStrategy):
    """Strategy that never retries.

    Use this when you want to disable retries for specific tools
    or in testing scenarios.

    Example:
        strategy = NoRetryStrategy()
        assert strategy.should_retry(context) is False
    """

    def should_retry(self, context: RetryContext) -> bool:
        """Always returns False - no retries."""
        return False

    def get_delay(self, context: RetryContext) -> float:
        """Returns 0 since we never retry."""
        return 0.0

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "NoRetryStrategy"


class FixedDelayStrategy(RetryStrategy):
    """Fixed delay between retries - simple and predictable.

    Each retry waits the same amount of time.

    Example:
        strategy = FixedDelayStrategy(max_retries=3, delay=5.0)
        # All retries wait 5 seconds
    """

    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        """Initialize fixed delay strategy.

        Args:
            max_retries: Maximum retry attempts
            delay: Fixed delay in seconds between retries
        """
        self.max_retries = max_retries
        self.delay = delay

    def should_retry(self, context: RetryContext) -> bool:
        """Check if should retry based on attempt count."""
        return context.attempt < self.max_retries

    def get_delay(self, context: RetryContext) -> float:
        """Return fixed delay."""
        return self.delay
