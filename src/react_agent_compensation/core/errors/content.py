"""Strategy for checking content dict patterns.

Checks for common error indicators in dict content, handling
both JSON responses and dict results.
"""

from __future__ import annotations

import json
from typing import Any

from react_agent_compensation.core.errors.base import ErrorStrategy


class ContentDictStrategy(ErrorStrategy):
    """Check content dict for common error indicators.

    Handles JSON and dict responses with structured error fields.
    Parses JSON strings if needed.

    Error patterns checked:
    - {"error": "message"} or {"error": {...}}
    - {"status": "error"} or {"status": "failed"}
    - {"success": false}
    - {"ok": false}
    - {"failed": true}

    Example:
        result.content = '{"error": "Not found"}'  -> Returns True
        result.content = {"status": "failed"}       -> Returns True
        result.content = {"success": false}         -> Returns True
    """

    ERROR_PATTERNS: list[tuple[str, list[Any]]] = [
        ("status", ["error", "failed", "failure"]),
        ("success", [False, "false", "False"]),
        ("ok", [False, "false", "False"]),
        ("failed", [True, "true", "True"]),
    ]

    def is_error(self, result: Any) -> bool | None:
        """Check content dict for error indicators.

        Args:
            result: The result to check

        Returns:
            True if error pattern found, False if success pattern, None otherwise
        """
        content = self._get_content(result)
        if content is None:
            return None

        # Parse JSON string to dict if needed
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                return None

        if not isinstance(content, dict):
            return None

        # Check for explicit error key
        if "error" in content and content["error"]:
            return True

        # Check common patterns
        for field, error_values in self.ERROR_PATTERNS:
            if field in content:
                value = content[field]
                if value in error_values:
                    return True
                # If field exists with non-error value, likely success
                if field in ("status", "success", "ok"):
                    if value not in error_values:
                        return False

        return None

    def _get_content(self, result: Any) -> Any:
        """Extract content from result.

        Args:
            result: The result object

        Returns:
            Content value (dict, string, or None)
        """
        # If result has content attribute, use it
        if hasattr(result, "content"):
            return result.content

        # If result is already a dict, use it directly
        if isinstance(result, dict):
            return result

        return None

    def get_error_message(self, result: Any) -> str | None:
        """Extract error message from content dict.

        Args:
            result: The result to extract message from

        Returns:
            Error message if found
        """
        content = self._get_content(result)
        if content is None:
            return None

        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                return None

        if not isinstance(content, dict):
            return None

        # Check common error message keys
        for key in ["error", "message", "error_message", "detail", "reason"]:
            if key in content:
                value = content[key]
                if isinstance(value, str):
                    return value
                if isinstance(value, dict) and "message" in value:
                    return str(value["message"])

        return None
