"""State synchronization for AWS Strands compensation.

This module provides utilities for synchronizing the compensation
TransactionLog and FailureContext with Strands' invocation_state,
enabling persistence across agent invocations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from react_agent_compensation.core.models import FailureContext
from react_agent_compensation.core.transaction_log import TransactionLog

if TYPE_CHECKING:
    from react_agent_compensation.core.recovery_manager import RecoveryManager


COMPENSATION_LOG_KEY = "compensation_log"
FAILURE_CONTEXT_KEY = "failure_context"


class StrandsStateSync:
    """Synchronizes TransactionLog and FailureContext with invocation_state.

    Strands provides invocation_state dict for persisting state across
    agent invocations. This class syncs compensation state with it.

    Use this for:
    - Persisting transaction log across agent invocations
    - Persisting failure context for Strategic Context Preservation
    - Integrating with Strands' state management

    Example:
        sync = StrandsStateSync()

        # In hook, load state from invocation_state
        sync.load_into_manager(event.invocation_state, middleware.rc_manager)

        # After execution, save state back
        sync.save_from_manager(event.invocation_state, middleware.rc_manager)
    """

    def __init__(
        self,
        log_key: str = COMPENSATION_LOG_KEY,
        failure_context_key: str = FAILURE_CONTEXT_KEY,
    ):
        """Initialize state sync.

        Args:
            log_key: Key in invocation_state for transaction log
            failure_context_key: Key in invocation_state for failure context
        """
        self.log_key = log_key
        self.failure_context_key = failure_context_key

    def load_into_manager(
        self,
        invocation_state: dict[str, Any],
        manager: "RecoveryManager",
    ) -> None:
        """Load state from invocation_state into RecoveryManager.

        Args:
            invocation_state: Strands invocation state dict
            manager: RecoveryManager to load state into
        """
        # Load transaction log
        log_data = invocation_state.get(self.log_key, {})
        if log_data:
            manager._log = TransactionLog.from_dict(log_data)

        # Load failure context
        ctx_data = invocation_state.get(self.failure_context_key, {})
        if ctx_data:
            manager._failure_context = FailureContext.model_validate(ctx_data)

    def save_from_manager(
        self,
        invocation_state: dict[str, Any],
        manager: "RecoveryManager",
    ) -> None:
        """Save state from RecoveryManager to invocation_state.

        Args:
            invocation_state: Strands invocation state dict
            manager: RecoveryManager to save state from
        """
        # Save transaction log
        invocation_state[self.log_key] = manager.log.to_dict()

        # Save failure context
        invocation_state[self.failure_context_key] = manager.failure_context.model_dump()

    def load_transaction_log(
        self,
        invocation_state: dict[str, Any],
    ) -> TransactionLog:
        """Load TransactionLog from invocation_state.

        Args:
            invocation_state: Strands invocation state dict

        Returns:
            TransactionLog instance
        """
        data = invocation_state.get(self.log_key, {})
        return TransactionLog.from_dict(data)

    def save_transaction_log(
        self,
        invocation_state: dict[str, Any],
        log: TransactionLog,
    ) -> None:
        """Save TransactionLog to invocation_state.

        Args:
            invocation_state: Strands invocation state dict
            log: TransactionLog to save
        """
        invocation_state[self.log_key] = log.to_dict()

    def load_failure_context(
        self,
        invocation_state: dict[str, Any],
    ) -> FailureContext:
        """Load FailureContext from invocation_state.

        Args:
            invocation_state: Strands invocation state dict

        Returns:
            FailureContext instance
        """
        data = invocation_state.get(self.failure_context_key, {})
        if data:
            return FailureContext.model_validate(data)
        return FailureContext()

    def save_failure_context(
        self,
        invocation_state: dict[str, Any],
        context: FailureContext,
    ) -> None:
        """Save FailureContext to invocation_state.

        Args:
            invocation_state: Strands invocation state dict
            context: FailureContext to save
        """
        invocation_state[self.failure_context_key] = context.model_dump()


def get_compensation_log(
    invocation_state: dict[str, Any],
    key: str = COMPENSATION_LOG_KEY,
) -> TransactionLog | None:
    """Get TransactionLog from invocation_state.

    Args:
        invocation_state: Strands invocation state dict
        key: Key where log is stored

    Returns:
        TransactionLog or None if not found
    """
    data = invocation_state.get(key)
    if data:
        return TransactionLog.from_dict(data)
    return None


def sync_compensation_log(
    invocation_state: dict[str, Any],
    log: TransactionLog,
    key: str = COMPENSATION_LOG_KEY,
) -> None:
    """Sync TransactionLog to invocation_state.

    Args:
        invocation_state: Strands invocation state dict
        log: TransactionLog to sync
        key: Key to use in state dict
    """
    invocation_state[key] = log.to_dict()


def get_failure_context(
    invocation_state: dict[str, Any],
    key: str = FAILURE_CONTEXT_KEY,
) -> FailureContext | None:
    """Get FailureContext from invocation_state.

    Args:
        invocation_state: Strands invocation state dict
        key: Key where failure context is stored

    Returns:
        FailureContext or None if not found
    """
    data = invocation_state.get(key)
    if data:
        return FailureContext.model_validate(data)
    return None


def sync_failure_context(
    invocation_state: dict[str, Any],
    context: FailureContext,
    key: str = FAILURE_CONTEXT_KEY,
) -> None:
    """Sync FailureContext to invocation_state.

    Args:
        invocation_state: Strands invocation state dict
        context: FailureContext to sync
        key: Key to use in state dict
    """
    invocation_state[key] = context.model_dump()
