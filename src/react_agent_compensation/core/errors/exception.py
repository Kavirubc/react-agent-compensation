"""Strategy for detecting error messages that look like exceptions.

Checks string content for patterns that indicate exceptions,
errors, or failures.
"""

from __future__ import annotations

from typing import Any

from react_agent_compensation.core.errors.base import ErrorStrategy


class ExceptionContentStrategy(ErrorStrategy):
    """Detect error messages that look like Python exceptions.

    Checks string content for patterns indicating exceptions or errors.
    Useful for tools that return error messages as strings.

    Patterns checked:
    - Strings starting with "error:", "exception:", "failed:", etc.
    - Python tracebacks
    - HTTP error codes in text

    Example:
        "Error: Connection refused"     -> Returns True
        "ValueError: invalid input"     -> Returns True
        "Traceback (most recent..."     -> Returns True
    """

    ERROR_PREFIXES = [
        "error:",
        "exception:",
        "failed:",
        "failure:",
        "valueerror:",
        "typeerror:",
        "keyerror:",
        "indexerror:",
        "attributeerror:",
        "runtimeerror:",
        "connectionerror:",
        "timeout:",
        "timeouterror:",
        "httperror:",
        "apierror:",
        "invalid:",
        "cannot ",
        "could not ",
        "unable to ",
        "not found:",
        "permission denied:",
        "access denied:",
        "unauthorized:",
        "forbidden:",
    ]

    HTTP_ERROR_PATTERNS = [
        "400 bad request",
        "401 unauthorized",
        "403 forbidden",
        "404 not found",
        "500 internal server error",
        "502 bad gateway",
        "503 service unavailable",
        "504 gateway timeout",
    ]

    def is_error(self, result: Any) -> bool | None:
        """Check if string content looks like an error.

        Args:
            result: The result to check

        Returns:
            True if error pattern found, None otherwise
        """
        content = self._get_string_content(result)
        if not content:
            return None

        content_lower = content.lower().strip()

        # Check for error prefixes
        for prefix in self.ERROR_PREFIXES:
            if content_lower.startswith(prefix):
                return True

        # Check for Python traceback
        if "traceback (most recent call last)" in content_lower:
            return True

        # Check for HTTP error patterns
        for pattern in self.HTTP_ERROR_PATTERNS:
            if pattern in content_lower:
                return True

        # Check for exception class patterns (e.g., "SomeError: message")
        if ": " in content and content.split(": ")[0].endswith("Error"):
            return True
        if ": " in content and content.split(": ")[0].endswith("Exception"):
            return True

        return None

    def _get_string_content(self, result: Any) -> str | None:
        """Extract string content from result.

        Args:
            result: The result object

        Returns:
            String content or None
        """
        # If result has content attribute
        if hasattr(result, "content"):
            content = result.content
            if isinstance(content, str):
                return content

        # If result is a string
        if isinstance(result, str):
            return result

        return None

    def get_error_message(self, result: Any) -> str | None:
        """Return the error string content.

        Args:
            result: The result to extract message from

        Returns:
            The error string content
        """
        content = self._get_string_content(result)
        if content and self.is_error(result):
            # Truncate very long error messages
            if len(content) > 500:
                return content[:500] + "..."
            return content
        return None
