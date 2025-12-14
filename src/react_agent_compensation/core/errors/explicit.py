"""Strategy for checking explicit status fields.

Most reliable error detection - checks for explicit status attributes
or fields that indicate success/failure.
"""

from __future__ import annotations

from typing import Any

from react_agent_compensation.core.errors.base import ErrorStrategy


class ExplicitStatusStrategy(ErrorStrategy):
    """Check for explicit status attributes on results.

    This is the most reliable strategy - checks for explicit status
    indicators that frameworks like LangChain provide.

    Checks (in order):
    1. result.status == "error"
    2. result.status == "success" (not an error)
    3. result.is_error attribute

    Example:
        # LangChain ToolMessage with status
        result.status = "error"  -> Returns True

        # Custom result with is_error flag
        result.is_error = True   -> Returns True
    """

    ERROR_STATUSES = {"error", "failed", "failure"}
    SUCCESS_STATUSES = {"success", "ok", "completed"}

    def is_error(self, result: Any) -> bool | None:
        """Check for explicit status attribute.

        Args:
            result: The result to check

        Returns:
            True if error status, False if success status, None otherwise
        """
        # Check for status attribute
        if hasattr(result, "status"):
            status = result.status
            if isinstance(status, str):
                status_lower = status.lower()
                if status_lower in self.ERROR_STATUSES:
                    return True
                if status_lower in self.SUCCESS_STATUSES:
                    return False

        # Check for is_error attribute
        if hasattr(result, "is_error"):
            return bool(result.is_error)

        # Check for success attribute (inverse)
        if hasattr(result, "success"):
            return not result.success

        return None

    def get_error_message(self, result: Any) -> str | None:
        """Extract error message from result.

        Args:
            result: The result to extract message from

        Returns:
            Error message if available
        """
        # Check common error message attributes
        for attr in ["error_message", "message", "error", "detail"]:
            if hasattr(result, attr):
                msg = getattr(result, attr)
                if isinstance(msg, str):
                    return msg

        return None
