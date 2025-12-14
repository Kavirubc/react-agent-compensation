"""Exponential backoff retry strategy with jitter.

Implements exponential backoff with optional jitter to prevent
thundering herd problems in distributed systems.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from react_agent_compensation.core.config import RetryPolicy
from react_agent_compensation.core.retry.base import RetryContext, RetryStrategy

if TYPE_CHECKING:
    pass


class ExponentialBackoffStrategy(RetryStrategy):
    """Implements exponential backoff with optional jitter.

    Delay formula: min(max_delay, initial_delay * (multiplier ^ (attempt - 1)))
    With jitter: delay * random(0.5, 1.5)

    Example:
        policy = RetryPolicy(
            max_retries=3,
            initial_delay=1.0,
            max_delay=30.0,
            backoff_multiplier=2.0,
            jitter=True,
        )
        strategy = ExponentialBackoffStrategy(policy)

        # Delays (approximate with jitter):
        # Attempt 1: ~1s
        # Attempt 2: ~2s
        # Attempt 3: ~4s
    """

    def __init__(self, policy: RetryPolicy | None = None):
        """Initialize with retry policy.

        Args:
            policy: RetryPolicy configuration. If None, uses defaults.
        """
        self.policy = policy or RetryPolicy()

    def should_retry(self, context: RetryContext) -> bool:
        """Determine if should retry based on attempt count and error type.

        Args:
            context: RetryContext with error and attempt information

        Returns:
            True if should retry, False otherwise
        """
        # Check if we've exceeded max retries
        if context.attempt >= self.policy.max_retries:
            return False

        # Check if the error is retryable
        error = context.error
        if isinstance(error, Exception):
            return self.policy.is_retryable_error(error)
        elif isinstance(error, str):
            return self.policy.is_retryable_error(error)

        return True

    def get_delay(self, context: RetryContext) -> float:
        """Calculate delay with exponential backoff and optional jitter.

        Args:
            context: RetryContext with error and attempt information

        Returns:
            Delay in seconds before next attempt
        """
        # Calculate base delay with exponential backoff
        delay = self.policy.initial_delay * (
            self.policy.backoff_multiplier ** (context.attempt - 1)
        )

        # Cap at max_delay
        delay = min(delay, self.policy.max_delay)

        # Apply jitter if enabled (randomize between 50% and 150%)
        if self.policy.jitter:
            jitter_factor = random.uniform(0.5, 1.5)
            delay *= jitter_factor

        return delay

    @property
    def name(self) -> str:
        """Return descriptive name with configuration."""
        return f"ExponentialBackoffStrategy(max_retries={self.policy.max_retries})"


class LinearBackoffStrategy(RetryStrategy):
    """Linear backoff strategy - delay increases linearly with each attempt.

    Simpler than exponential, useful when you want predictable delays.

    Example:
        strategy = LinearBackoffStrategy(
            max_retries=3,
            delay_increment=2.0,  # Add 2s per attempt
        )
        # Attempt 1: 2s, Attempt 2: 4s, Attempt 3: 6s
    """

    def __init__(
        self,
        max_retries: int = 3,
        delay_increment: float = 1.0,
        max_delay: float = 30.0,
        jitter: bool = True,
    ):
        """Initialize linear backoff strategy.

        Args:
            max_retries: Maximum retry attempts
            delay_increment: Seconds to add per attempt
            max_delay: Maximum delay cap
            jitter: Add randomness to delay
        """
        self.max_retries = max_retries
        self.delay_increment = delay_increment
        self.max_delay = max_delay
        self.jitter = jitter

    def should_retry(self, context: RetryContext) -> bool:
        """Check if should retry based on attempt count."""
        return context.attempt < self.max_retries

    def get_delay(self, context: RetryContext) -> float:
        """Calculate linear delay."""
        delay = self.delay_increment * context.attempt
        delay = min(delay, self.max_delay)

        if self.jitter:
            delay *= random.uniform(0.8, 1.2)

        return delay
