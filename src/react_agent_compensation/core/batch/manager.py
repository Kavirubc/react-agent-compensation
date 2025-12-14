"""Batch manager for coordinating parallel execution control.

Orchestrates batch detection, context management, and sequential
execution lock for proper handling of parallel tool calls.
"""

from __future__ import annotations

import threading
from typing import Any

from react_agent_compensation.core.batch.context import BatchContext, IntentDAG
from react_agent_compensation.core.batch.detector import BatchDetector
from react_agent_compensation.core.batch.lock import SequentialExecutionLock


class BatchManager:
    """Manages batch contexts and intent DAGs for parallel execution.

    Coordinates:
    - BatchDetector: Detects when parallel execution is happening
    - BatchContext: Tracks state for each parallel batch
    - IntentDAG: Records intended vs actual execution
    - SequentialExecutionLock: Forces sequential execution if enabled

    Example:
        manager = BatchManager(
            time_window_ms=50,
            track_intent=True,
            sequential_execution=True,
        )

        # Record a call and detect batch
        batch_id = manager.detect_batch("tool1", thread_id, tool_call_id)

        # Get sequential lock for ordered execution
        lock = manager.get_sequential_lock()
        if lock:
            with lock.acquire_execution_slot(tool_call_id) as slot:
                if slot.should_abort:
                    return aborted_result
                # Execute tool...
    """

    def __init__(
        self,
        time_window_ms: float = 50,
        track_intent: bool = False,
        sequential_execution: bool = False,
    ):
        """Initialize batch manager.

        Args:
            time_window_ms: Time window for batch detection
            track_intent: Whether to track intended vs actual execution
            sequential_execution: Whether to force sequential execution
        """
        self._detector = BatchDetector(time_window_ms=time_window_ms)
        self._contexts: dict[str, BatchContext] = {}
        self._intent_dags: dict[str, IntentDAG] = {}
        self._sequential_lock: SequentialExecutionLock | None = (
            SequentialExecutionLock() if sequential_execution else None
        )
        self._track_intent = track_intent
        self._lock = threading.Lock()

    def detect_batch(
        self, tool_name: str, thread_id: str, tool_call_id: str
    ) -> str | None:
        """Detect if call is part of parallel batch.

        Args:
            tool_name: Name of the tool
            thread_id: ID of the executing thread
            tool_call_id: Unique tool call ID

        Returns:
            batch_id if parallel batch detected, None otherwise
        """
        return self._detector.record_call(tool_name, thread_id, tool_call_id)

    def get_or_create_context(
        self, batch_id: str, tool_count: int, tool_call_ids: list[str]
    ) -> BatchContext:
        """Get or create batch context.

        Args:
            batch_id: ID of the batch
            tool_count: Number of tools in batch
            tool_call_ids: List of tool call IDs

        Returns:
            BatchContext for the batch
        """
        with self._lock:
            if batch_id not in self._contexts:
                self._contexts[batch_id] = BatchContext(
                    batch_id=batch_id,
                    tool_count=tool_count,
                    tool_call_ids=tool_call_ids,
                )
            return self._contexts[batch_id]

    def get_context(self, batch_id: str) -> BatchContext | None:
        """Get existing batch context.

        Args:
            batch_id: ID of the batch

        Returns:
            BatchContext or None if not found
        """
        with self._lock:
            return self._contexts.get(batch_id)

    def get_or_create_intent_dag(self, batch_id: str) -> IntentDAG | None:
        """Get or create intent DAG for batch.

        Args:
            batch_id: ID of the batch

        Returns:
            IntentDAG or None if tracking disabled
        """
        if not self._track_intent:
            return None

        with self._lock:
            if batch_id not in self._intent_dags:
                self._intent_dags[batch_id] = IntentDAG(batch_id)
            return self._intent_dags[batch_id]

    def get_sequential_lock(self) -> SequentialExecutionLock | None:
        """Get sequential execution lock if enabled.

        Returns:
            SequentialExecutionLock or None if not enabled
        """
        return self._sequential_lock

    def reset_sequential_lock(self) -> None:
        """Reset sequential execution lock for new agent turn."""
        if self._sequential_lock:
            self._sequential_lock.reset()

    def cleanup_batch(self, batch_id: str) -> dict[str, Any] | None:
        """Clean up completed batch and return report.

        Args:
            batch_id: ID of the batch to clean up

        Returns:
            Intent DAG report if tracking was enabled, None otherwise
        """
        with self._lock:
            report = None

            # Get intent report before cleanup
            if batch_id in self._intent_dags:
                report = self._intent_dags[batch_id].get_report()
                del self._intent_dags[batch_id]

            # Clean up context
            if batch_id in self._contexts:
                del self._contexts[batch_id]

            # Clean up detector
            self._detector.cleanup_batch(batch_id)

            return report

    def signal_abort(
        self, batch_id: str, tool_name: str, tool_call_id: str, reason: str
    ) -> bool:
        """Signal abort for a batch.

        Signals both batch context and sequential lock.

        Args:
            batch_id: ID of the batch
            tool_name: Name of the failed tool
            tool_call_id: ID of the failed tool call
            reason: Reason for abort

        Returns:
            True if abort was signaled, False if already aborted
        """
        signaled = False

        # Signal batch context
        context = self.get_context(batch_id)
        if context:
            signaled = context.signal_abort(tool_name, tool_call_id, reason)

        # Signal sequential lock
        if self._sequential_lock:
            self._sequential_lock.signal_abort(tool_name, reason)

        # Abort pending intents
        dag = self._intent_dags.get(batch_id) if self._track_intent else None
        if dag:
            dag.abort_pending()

        return signaled

    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self._contexts.clear()
            self._intent_dags.clear()
            self._detector.reset()
            if self._sequential_lock:
                self._sequential_lock.reset()
