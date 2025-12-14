"""Exception classes for react-agent-compensation.

This module defines custom exceptions used throughout the library:
- RollbackFailure: Raised when a rollback operation fails
- CriticalFailure: Raised when system enters inconsistent state
- ExtractionError: Raised when parameter extraction fails
- RecoveryError: Raised when recovery (retry/alternatives) fails
"""

from __future__ import annotations

from typing import Any


class CompensationError(Exception):
    """Base exception for all compensation-related errors."""

    pass


class RollbackFailure(CompensationError):
    """Raised when a rollback operation fails.

    This indicates that one or more compensation actions could not be executed,
    but the system may still be in a recoverable state.

    Attributes:
        message: Description of the failure
        failed_records: List of record IDs that failed to compensate
        compensated_records: List of record IDs that were successfully compensated
    """

    def __init__(
        self,
        message: str,
        failed_records: list[str] | None = None,
        compensated_records: list[str] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.failed_records = failed_records or []
        self.compensated_records = compensated_records or []


class CriticalFailure(CompensationError):
    """Raised when the system enters an inconsistent state.

    This is a severe error indicating that compensation failed in a way
    that leaves the system in an unknown state. Manual intervention may
    be required.

    Attributes:
        message: Description of the critical failure
        context: Additional context about the failure
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}


class ExtractionError(CompensationError):
    """Raised when parameter extraction fails.

    This indicates that the system could not extract the necessary
    parameters to execute a compensation action.

    Attributes:
        message: Description of the extraction failure
        tool_name: Name of the tool being compensated
        result: The result that failed extraction
    """

    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        result: Any = None,
    ):
        super().__init__(message)
        self.message = message
        self.tool_name = tool_name
        self.result = result


class RecoveryError(CompensationError):
    """Raised when recovery (retry/alternatives) fails.

    This indicates that all recovery attempts have been exhausted
    and the operation cannot be completed.

    Attributes:
        message: Description of the recovery failure
        attempts: Number of attempts made
        last_error: The last error encountered
    """

    def __init__(
        self,
        message: str,
        attempts: int = 0,
        last_error: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.attempts = attempts
        self.last_error = last_error
