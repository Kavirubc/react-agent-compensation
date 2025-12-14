"""Configuration models for react-agent-compensation.

This module defines configuration structures:
- RetryPolicy: Configuration for retry behavior
- CompensationPairs: Type alias for tool-to-compensation mappings
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# Type alias for compensation mappings: {"forward_tool": "undo_tool"}
CompensationPairs = dict[str, str]

# Type alias for alternative mappings: {"tool": ["alt1", "alt2"]}
AlternativeMap = dict[str, list[str]]


class RetryPolicy(BaseModel):
    """Configuration for retry behavior on transient failures.

    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries)
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay cap in seconds
        backoff_multiplier: Multiplier for exponential backoff
        jitter: Add randomness to delay to prevent thundering herd
        retryable_errors: List of error types/patterns that should trigger retry

    Example:
        policy = RetryPolicy(
            max_retries=3,
            initial_delay=1.0,
            max_delay=30.0,
            backoff_multiplier=2.0,
            jitter=True,
        )
    """

    max_retries: int = Field(default=3, ge=0)
    initial_delay: float = Field(default=1.0, gt=0)
    max_delay: float = Field(default=60.0, gt=0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    jitter: bool = True
    retryable_errors: list[str] = Field(
        default_factory=lambda: ["timeout", "connection", "rate_limit"]
    )

    def is_retryable_error(self, error: Exception | str) -> bool:
        """Check if an error matches retryable patterns."""
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in self.retryable_errors)


class RecoveryConfig(BaseModel):
    """Complete configuration for the recovery system.

    Attributes:
        compensation_pairs: Mapping of tools to their compensation tools
        alternative_map: Mapping of tools to alternative tools to try
        retry_policy: Configuration for retry behavior
        partial_rollback: If True, only rollback dependent actions
        infer_dependencies: If True, auto-detect dependencies via data flow
    """

    compensation_pairs: CompensationPairs = Field(default_factory=dict)
    alternative_map: AlternativeMap = Field(default_factory=dict)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    partial_rollback: bool = False
    infer_dependencies: bool = True
