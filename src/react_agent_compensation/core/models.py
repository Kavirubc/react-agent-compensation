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
