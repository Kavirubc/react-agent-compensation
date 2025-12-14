"""Base classes for retry strategies.

This module defines:
- RetryStrategy: Abstract base class for all retry strategies
- RetryContext: Context information for retry decisions
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetryContext:
    """Context information for retry decisions.

    Attributes:
        error: The exception or error that occurred
        attempt: Current attempt number (1-indexed)
        action: Name of the action that failed
        params: Parameters passed to the action
        elapsed_time: Total time elapsed since first attempt (seconds)
    """

    error: Exception | str
    attempt: int
    action: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    elapsed_time: float = 0.0


class RetryStrategy(ABC):
    """Abstract base class for retry strategies.

    Determines whether to retry after a failure and how long to wait.
    Implementations decide based on error type, attempt count, etc.

    Example:
        class MyRetryStrategy(RetryStrategy):
            def should_retry(self, context: RetryContext) -> bool:
                return context.attempt < 3

            def get_delay(self, context: RetryContext) -> float:
                return 1.0 * context.attempt
    """

    @abstractmethod
    def should_retry(self, context: RetryContext) -> bool:
        """Determine if the operation should be retried.

        Args:
            context: RetryContext with error and attempt information

        Returns:
            True if should retry, False otherwise
        """
        pass

    @abstractmethod
    def get_delay(self, context: RetryContext) -> float:
        """Calculate delay before next retry.

        Args:
            context: RetryContext with error and attempt information

        Returns:
            Delay in seconds before next attempt
        """
        pass

    @property
    def name(self) -> str:
        """Human-readable name for logging and debugging."""
        return self.__class__.__name__

    def on_retry(self, context: RetryContext) -> None:
        """Hook called before each retry attempt.

        Override to add logging, metrics, etc.

        Args:
            context: RetryContext with error and attempt information
        """
        pass

    def on_exhausted(self, context: RetryContext) -> None:
        """Hook called when all retries are exhausted.

        Override to add logging, metrics, cleanup, etc.

        Args:
            context: RetryContext with error and attempt information
        """
        pass
