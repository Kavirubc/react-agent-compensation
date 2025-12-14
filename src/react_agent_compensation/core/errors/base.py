"""Base classes for error detection strategies.

This module defines:
- ErrorStrategy: Abstract base class for all error detection strategies
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ErrorStrategy(ABC):
    """Abstract base class for error detection strategies.

    Each strategy attempts to determine if a result indicates an error.
    Strategies can:
    - Return True: Result is definitely an error
    - Return False: Result is definitely not an error
    - Return None: Cannot determine, defer to next strategy

    This allows strategies to be chained for flexible error detection.
    """

    @abstractmethod
    def is_error(self, result: Any) -> bool | None:
        """Determine if the result indicates an error.

        Args:
            result: The result to check (could be dict, string, object, etc.)

        Returns:
            True if error, False if not error, None if cannot determine
        """
        pass

    @property
    def name(self) -> str:
        """Human-readable name for logging and debugging."""
        return self.__class__.__name__

    def get_error_message(self, result: Any) -> str | None:
        """Extract error message from result if available.

        Override to provide custom error message extraction.

        Args:
            result: The result to extract message from

        Returns:
            Error message string, or None if not available
        """
        return None
