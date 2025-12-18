"""Heuristics for detecting permanent vs transient failures.

This module provides pattern-based detection to classify errors as either:
- Permanent: Won't succeed on retry (e.g., "not found", "unauthorized")
- Transient: May succeed on retry (e.g., "timeout", "rate limit")

The classification helps the LLM understand which failed attempts are
worth retrying with the same parameters vs. which need a different approach.
"""

from __future__ import annotations

import re

# Patterns that suggest permanent failures (won't succeed on retry)
PERMANENT_PATTERNS = [
    r"unavailable",
    r"not found",
    r"does not exist",
    r"invalid",
    r"unauthorized",
    r"forbidden",
    r"permanently",
    r"disabled",
    r"deprecated",
    r"removed",
    r"breakdown",
    r"broken",
    r"not supported",
    r"no longer available",
    r"deleted",
    r"terminated",
    r"cancelled",
    r"rejected",
]

# Patterns that suggest transient failures (may succeed on retry)
TRANSIENT_PATTERNS = [
    r"timeout",
    r"timed out",
    r"temporarily",
    r"rate.?limit",
    r"too many requests",
    r"busy",
    r"overloaded",
    r"retry",
    r"connection.?refused",
    r"connection.?reset",
    r"network",
    r"503",
    r"502",
    r"504",
    r"service unavailable",
    r"try again",
]


def is_likely_permanent(error: str) -> bool:
    """Heuristic to determine if an error is likely permanent.

    Uses pattern matching against known permanent and transient error patterns.
    Transient patterns take precedence (if both match, assumes transient).

    Args:
        error: Error message to analyze

    Returns:
        True if the error appears permanent (won't succeed on retry),
        False if transient or unknown (may succeed on retry)

    Example:
        >>> is_likely_permanent("Machine unavailable due to breakdown")
        True
        >>> is_likely_permanent("Request timed out, please retry")
        False
        >>> is_likely_permanent("Something went wrong")
        False
    """
    error_lower = error.lower()

    # Check for transient patterns first (they override permanent)
    for pattern in TRANSIENT_PATTERNS:
        if re.search(pattern, error_lower):
            return False

    # Check for permanent patterns
    for pattern in PERMANENT_PATTERNS:
        if re.search(pattern, error_lower):
            return True

    # Default: unknown, assume not permanent (allow retry)
    return False
