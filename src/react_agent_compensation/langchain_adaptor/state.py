"""LangGraph state synchronization for TransactionLog.

This module provides utilities for synchronizing the compensation
TransactionLog with LangGraph state, enabling persistence and
multi-agent coordination.
"""

from __future__ import annotations

from typing import Any

from react_agent_compensation.core.transaction_log import TransactionLog


ACTION_LOG_KEY = "compensation_log"


class LangGraphStateSync:
    """Synchronizes TransactionLog with LangGraph state.

    Use this for:
    - Persisting transaction log across graph executions
    - Sharing log between multiple agents in a multi-agent graph
    - Integrating with LangGraph checkpointing

    Example:
        sync = LangGraphStateSync()

        # Before tool execution
        log = sync.load(state)
        middleware.rc_manager._log = log

        # After tool execution
        sync.save(state, middleware.transaction_log)
    """

    def __init__(self, state_key: str = ACTION_LOG_KEY):
        """Initialize state sync.

        Args:
            state_key: Key to use in state dict for the log
        """
        self.state_key = state_key

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
