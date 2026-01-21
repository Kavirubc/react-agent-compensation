"""State management for CrewAI compensation.

This module provides utilities for synchronizing the compensation
TransactionLog and FailureContext with CrewAI's execution context,
enabling persistence and multi-agent coordination.
"""

from __future__ import annotations

from typing import Any

from react_agent_compensation.core.models import FailureContext
from react_agent_compensation.core.transaction_log import TransactionLog


COMPENSATION_LOG_KEY = "compensation_log"
FAILURE_CONTEXT_KEY = "failure_context"


class CrewAIStateSync:
    """Synchronizes TransactionLog and FailureContext with crew state.

    Use this for:
    - Persisting transaction log across crew executions
    - Persisting failure context for Strategic Context Preservation
    - Sharing log between multiple agents in a crew
    - Manual state management when using hooks

    Example:
        sync = CrewAIStateSync()

        # Save state after execution
        state = {}
        sync.save(state, middleware.transaction_log)
        sync.save_failure_context(state, middleware.rc_manager.failure_context)

        # Load state for next execution
        log = sync.load(state)
        failure_ctx = sync.load_failure_context(state)
    """

    def __init__(
        self,
        log_key: str = COMPENSATION_LOG_KEY,
        failure_context_key: str = FAILURE_CONTEXT_KEY,
    ):
        """Initialize state sync.

        Args:
            log_key: Key to use in state dict for the log
            failure_context_key: Key to use in state dict for failure context
        """
        self.log_key = log_key
        self.failure_context_key = failure_context_key

    def load(self, state: dict[str, Any]) -> TransactionLog:
        """Load TransactionLog from state dict.

        Args:
            state: State dictionary

        Returns:
            TransactionLog instance (new or restored)
        """
        data = state.get(self.log_key, {})
        return TransactionLog.from_dict(data)

    def save(self, state: dict[str, Any], log: TransactionLog) -> None:
        """Save TransactionLog to state dict.

        Args:
            state: State dictionary
            log: TransactionLog to save
        """
        state[self.log_key] = log.to_dict()

    def load_failure_context(self, state: dict[str, Any]) -> FailureContext:
        """Load FailureContext from state dict.

        Args:
            state: State dictionary

        Returns:
            FailureContext instance (new or restored)
        """
        data = state.get(self.failure_context_key, {})
        if data:
            return FailureContext.model_validate(data)
        return FailureContext()

    def save_failure_context(
        self, state: dict[str, Any], context: FailureContext
    ) -> None:
        """Save FailureContext to state dict.

        Args:
            state: State dictionary
            context: FailureContext to save
        """
        state[self.failure_context_key] = context.model_dump()

    def merge_logs(
        self,
        state: dict[str, Any],
        log: TransactionLog,
        agent_id: str | None = None,
    ) -> TransactionLog:
        """Merge local log with state log.

        Useful for multi-agent scenarios where each agent has its own
        local log but they share a common state.

        Args:
            state: State dictionary
            log: Local TransactionLog to merge
            agent_id: Only merge records from this agent

        Returns:
            Merged TransactionLog
        """
        existing = self.load(state)
        local_records = log.snapshot()

        for record_id, record in local_records.items():
            if agent_id and record.agent_id != agent_id:
                continue
            existing_record = existing.get(record_id)
            if existing_record is None:
                existing.add(record)

        return existing


def get_compensation_log(
    state: dict[str, Any],
    key: str = COMPENSATION_LOG_KEY,
) -> TransactionLog | None:
    """Get TransactionLog from state dictionary.

    Args:
        state: State dictionary
        key: Key where log is stored

    Returns:
        TransactionLog or None if not found
    """
    data = state.get(key)
    if data:
        return TransactionLog.from_dict(data)
    return None


def sync_compensation_log(
    state: dict[str, Any],
    log: TransactionLog,
    key: str = COMPENSATION_LOG_KEY,
) -> None:
    """Sync TransactionLog to state dictionary.

    Args:
        state: State dictionary
        log: TransactionLog to sync
        key: Key to use in state dict
    """
    state[key] = log.to_dict()


def get_failure_context(
    state: dict[str, Any],
    key: str = FAILURE_CONTEXT_KEY,
) -> FailureContext | None:
    """Get FailureContext from state dictionary.

    Args:
        state: State dictionary
        key: Key where failure context is stored

    Returns:
        FailureContext or None if not found
    """
    data = state.get(key)
    if data:
        return FailureContext.model_validate(data)
    return None


def sync_failure_context(
    state: dict[str, Any],
    context: FailureContext,
    key: str = FAILURE_CONTEXT_KEY,
) -> None:
    """Sync FailureContext to state dictionary.

    Args:
        state: State dictionary
        context: FailureContext to sync
        key: Key to use in state dict
    """
    state[key] = context.model_dump()


def create_shared_state() -> dict[str, Any]:
    """Create a shared state dictionary for multi-agent scenarios.

    Returns:
        Empty state dictionary ready for compensation tracking
    """
    return {}
