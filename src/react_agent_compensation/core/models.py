"""Core data models for react-agent-compensation.

This module defines the fundamental data structures used throughout the library:
- ActionStatus: Enum representing the lifecycle states of an action
- ActionRecord: Pydantic model tracking a single compensatable action
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ActionStatus(str, Enum):
    """Lifecycle status of an action in the compensation system.

    States:
        PENDING: Action recorded but not yet executed
        COMPLETED: Action executed successfully
        FAILED: Action execution failed
        COMPENSATED: Action was rolled back via compensation
    """

    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    COMPENSATED = "COMPENSATED"


class ActionRecord(BaseModel):
    """Record of a single compensatable action.

    Tracks all information needed to:
    - Execute compensation (rollback) if needed
    - Determine rollback order via dependencies
    - Support multi-agent scenarios via agent_id

    Attributes:
        id: Unique identifier for this action record
        action: Name of the tool/action executed
        params: Parameters passed to the action
        result: Result returned by the action (set after completion)
        status: Current lifecycle status
        compensator: Name of the compensation tool to call for rollback
        depends_on: List of action IDs this action depends on
        timestamp: Unix timestamp when the action was recorded
        agent_id: Identifier for multi-agent scenarios
        compensated: Whether compensation has been executed
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    action: str
    params: dict[str, Any]
    result: Any = None
    status: ActionStatus = ActionStatus.PENDING
    compensator: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)
    agent_id: str | None = None
    compensated: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def mark_completed(self, result: Any) -> None:
        """Mark this action as successfully completed."""
        self.status = ActionStatus.COMPLETED
        self.result = result

    def mark_failed(self, error: str | None = None) -> None:
        """Mark this action as failed."""
        self.status = ActionStatus.FAILED
        if error:
            self.result = {"error": error}

    def mark_compensated(self) -> None:
        """Mark this action as compensated (rolled back)."""
        self.status = ActionStatus.COMPENSATED
        self.compensated = True

    def is_compensatable(self) -> bool:
        """Check if this action can be compensated."""
        return (
            self.status == ActionStatus.COMPLETED
            and not self.compensated
            and self.compensator is not None
        )


class FailedAttempt(BaseModel):
    """Record of a failed action attempt for Strategic Context Preservation.

    Domain-agnostic: tracks what was tried (action + params), not domain-specific
    resources. This enables the LLM to learn from failures across ANY problem domain.

    Attributes:
        action: Name of the tool/action that failed
        params: Parameters that were used in the attempt
        error: Error message from the failure
        timestamp: Unix timestamp when the failure occurred
        attempt_number: Which attempt this was for this action
        is_permanent: Heuristic flag indicating if failure is likely permanent
    """

    action: str
    params: dict[str, Any]
    error: str
    timestamp: float = Field(default_factory=time.time)
    attempt_number: int = 1
    is_permanent: bool = False

    def params_signature(self) -> str:
        """Generate a hashable signature of params for deduplication.

        Returns:
            String representation of sorted params for comparison
        """
        return str(sorted(self.params.items()))


class FailureContext(BaseModel):
    """Tracks cumulative failures across retries - domain agnostic.

    Provides the LLM with context about what has been tried and failed,
    enabling it to make informed decisions about what to try next.
    This is the core of Strategic Context Preservation.

    Unlike domain-specific approaches (e.g., tracking "broken machines"),
    this tracks action+params combinations that failed, working for ANY domain.

    Attributes:
        attempts: List of all failed attempts in chronological order
    """

    attempts: list[FailedAttempt] = Field(default_factory=list)

    def record_attempt(
        self,
        action: str,
        params: dict[str, Any],
        error: str,
        is_permanent: bool = False,
    ) -> None:
        """Record a failed attempt.

        Args:
            action: Name of the action/tool that failed
            params: Parameters that were used
            error: Error message from the failure
            is_permanent: Whether the failure appears permanent
        """
        attempt_num = sum(1 for a in self.attempts if a.action == action) + 1
        self.attempts.append(
            FailedAttempt(
                action=action,
                params=params,
                error=error,
                attempt_number=attempt_num,
                is_permanent=is_permanent,
            )
        )

    def get_attempts_for_action(self, action: str) -> list[FailedAttempt]:
        """Get all failed attempts for a specific action.

        Args:
            action: Name of the action to filter by

        Returns:
            List of FailedAttempt objects for that action
        """
        return [a for a in self.attempts if a.action == action]

    def has_similar_attempt(self, action: str, params: dict[str, Any]) -> bool:
        """Check if a similar attempt (same action + params) has failed before.

        Args:
            action: Name of the action
            params: Parameters to check

        Returns:
            True if this exact action+params combination failed before
        """
        sig = str(sorted(params.items()))
        return any(
            a.action == action and a.params_signature() == sig for a in self.attempts
        )

    def get_summary(self) -> str:
        """Generate human-readable summary for LLM context.

        Returns:
            Formatted string describing cumulative failures for LLM to use
            in its decision-making. Empty string if no failures recorded.
        """
        if not self.attempts:
            return ""

        lines = ["[PREVIOUS FAILED ATTEMPTS]"]

        # Group by action
        by_action: dict[str, list[FailedAttempt]] = {}
        for attempt in self.attempts:
            by_action.setdefault(attempt.action, []).append(attempt)

        for action, attempts in by_action.items():
            lines.append(f"\n{action}:")
            for a in attempts:
                # Format params concisely
                param_str = ", ".join(f"{k}={v}" for k, v in a.params.items())
                permanent_marker = " [PERMANENT]" if a.is_permanent else ""
                lines.append(
                    f"  - Attempt {a.attempt_number}: ({param_str}){permanent_marker}"
                )
                # Truncate long errors
                error_preview = a.error[:100] + "..." if len(a.error) > 100 else a.error
                lines.append(f"    Error: {error_preview}")

        lines.append("\nConsider using different parameters or approaches.")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all recorded attempts."""
        self.attempts.clear()
