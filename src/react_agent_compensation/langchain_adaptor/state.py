"""LangGraph state synchronization for TransactionLog and FailureContext.

This module provides utilities for synchronizing the compensation
TransactionLog and FailureContext with LangGraph state, enabling
persistence and multi-agent coordination.

FailureContext is used for Strategic Context Preservation - tracking
cumulative failures across retries to help the LLM make informed decisions.
"""

from __future__ import annotations

from typing import Any

from react_agent_compensation.core.models import FailureContext
from react_agent_compensation.core.transaction_log import TransactionLog


ACTION_LOG_KEY = "compensation_log"
FAILURE_CONTEXT_KEY = "failure_context"


class LangGraphStateSync:
    """Synchronizes TransactionLog and FailureContext with LangGraph state.

    Use this for:
    - Persisting transaction log across graph executions
    - Persisting failure context for Strategic Context Preservation
    - Sharing log between multiple agents in a multi-agent graph
    - Integrating with LangGraph checkpointing

    Example:
        sync = LangGraphStateSync()

        # Before tool execution
        log = sync.load(state)
        failure_ctx = sync.load_failure_context(state)
        middleware.rc_manager._log = log
        middleware.rc_manager._failure_context = failure_ctx

        # After tool execution
        sync.save(state, middleware.transaction_log)
        sync.save_failure_context(state, middleware.rc_manager.failure_context)
    """

    def __init__(
        self,
        state_key: str = ACTION_LOG_KEY,
        failure_context_key: str = FAILURE_CONTEXT_KEY,
    ):
        """Initialize state sync.

        Args:
            state_key: Key to use in state dict for the log
            failure_context_key: Key to use in state dict for failure context
        """
        self.state_key = state_key
        self.failure_context_key = failure_context_key

    def load(self, state: dict[str, Any]) -> TransactionLog:
        """Load TransactionLog from state dict.

        Args:
            state: LangGraph state dict

        Returns:
            TransactionLog instance (new or restored)
        """
        data = state.get(self.state_key, {})
        return TransactionLog.from_dict(data)

    def save(self, state: dict[str, Any], log: TransactionLog) -> None:
        """Save TransactionLog to state dict.

        Args:
            state: LangGraph state dict
            log: TransactionLog to save
        """
        state[self.state_key] = log.to_dict()

    def load_failure_context(self, state: dict[str, Any]) -> FailureContext:
        """Load FailureContext from state dict.

        Args:
            state: LangGraph state dict

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
            state: LangGraph state dict
            context: FailureContext to save
        """
        state[self.failure_context_key] = context.model_dump()

    def merge(
        self,
        state: dict[str, Any],
        log: TransactionLog,
        agent_id: str | None = None,
    ) -> TransactionLog:
        """Merge local log with state log.

        Useful for multi-agent scenarios where each agent has its own
        local log but they share a common state.

        Args:
            state: LangGraph state dict
            log: Local TransactionLog to merge
            agent_id: Only merge records from this agent

        Returns:
            Merged TransactionLog
        """
        existing = self.load(state)

        # Get snapshot of local log
        local_records = log.snapshot()

        # Merge records
        for record_id, record in local_records.items():
            if agent_id and record.agent_id != agent_id:
                continue
            # Add or update record in existing log
            existing_record = existing.get(record_id)
            if existing_record is None:
                existing.add(record)

        return existing


def get_action_log(
    state: dict[str, Any],
    key: str = ACTION_LOG_KEY,
) -> TransactionLog | None:
    """Get TransactionLog from LangGraph state.

    Args:
        state: LangGraph state dict
        key: Key where log is stored

    Returns:
        TransactionLog or None if not found
    """
    data = state.get(key)
    if data:
        return TransactionLog.from_dict(data)
    return None


def sync_action_log(
    state: dict[str, Any],
    log: TransactionLog,
    key: str = ACTION_LOG_KEY,
) -> None:
    """Sync TransactionLog to LangGraph state.

    Args:
        state: LangGraph state dict
        log: TransactionLog to sync
        key: Key to use in state dict
    """
    state[key] = log.to_dict()


def create_shared_log() -> TransactionLog:
    """Create a shared TransactionLog for multi-agent scenarios.

    Returns:
        New TransactionLog instance
    """
    return TransactionLog()


def get_failure_context(
    state: dict[str, Any],
    key: str = FAILURE_CONTEXT_KEY,
) -> FailureContext | None:
    """Get FailureContext from LangGraph state.

    Args:
        state: LangGraph state dict
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
    """Sync FailureContext to LangGraph state.

    Args:
        state: LangGraph state dict
        context: FailureContext to sync
        key: Key to use in state dict
    """
    state[key] = context.model_dump()
