"""Sequential execution lock for compensatable tools.

Forces sequential execution of compensatable tools to ensure
proper ordering and abort handling.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionSlot:
    """Context for a single tool execution slot.

    Returned by SequentialExecutionLock.acquire_execution_slot().
    """

    tool_call_id: str
    transaction_id: str
    should_abort: bool = False
    abort_reason: str | None = None


class SequentialExecutionLock:
    """Forces sequential execution of compensatable tools.

    Ensures that parallel tool calls from the LLM execute one at a time,
    maintaining proper ordering for compensation. Includes abort handling
    so that if one tool fails, subsequent tools can be skipped.

    Example:
        lock = SequentialExecutionLock()

        # Tool 1 acquires slot
        with lock.acquire_execution_slot("tc-1") as slot:
            if slot.should_abort:
                return aborted_result
            # Execute tool 1...

        # Tool 2 acquires slot (blocked until tool 1 releases)
        with lock.acquire_execution_slot("tc-2") as slot:
            if slot.should_abort:
                return aborted_result
            # Execute tool 2...
    """

    def __init__(self):
        """Initialize sequential execution lock."""
        self._execution_lock = threading.Lock()
        self._abort_flag = threading.Event()
        self._abort_reason: str | None = None
        self._failed_tool: str | None = None

        # Transaction tracking
        self._transaction_id: str | None = None
        self._pending_count: int = 0
        self._entered_ids: set[str] = set()
        self._exited_ids: set[str] = set()
        self._state_lock = threading.Lock()

    def should_abort(self) -> bool:
        """Check if execution should abort (lock-free)."""
        return self._abort_flag.is_set()

    def signal_abort(self, tool_name: str, reason: str) -> None:
        """Signal that execution should abort.

        Args:
            tool_name: Name of the tool that failed
            reason: Reason for abort
        """
        with self._state_lock:
            if not self._abort_flag.is_set():
                self._failed_tool = tool_name
                self._abort_reason = reason
                self._abort_flag.set()

    def get_abort_info(self) -> tuple[str | None, str | None]:
        """Get abort information.

        Returns:
            Tuple of (failed_tool, abort_reason)
        """
        return self._failed_tool, self._abort_reason

    def enter_transaction(self, tool_call_id: str) -> str:
        """Record entry into transaction (before acquiring lock).

        Args:
            tool_call_id: ID of the tool call entering

        Returns:
            Transaction ID for this batch
        """
        with self._state_lock:
            if self._transaction_id is None:
                self._transaction_id = str(uuid.uuid4())

            if tool_call_id not in self._entered_ids:
                self._entered_ids.add(tool_call_id)
                self._pending_count += 1

            return self._transaction_id

    def exit_transaction(self, tool_call_id: str) -> bool:
        """Record exit from transaction (after execution).

        Args:
            tool_call_id: ID of the tool call exiting

        Returns:
            True if this was the last tool in the transaction
        """
        with self._state_lock:
            if tool_call_id in self._entered_ids and tool_call_id not in self._exited_ids:
                self._exited_ids.add(tool_call_id)
                self._pending_count -= 1

            is_last = self._pending_count <= 0
            if is_last:
                self._auto_reset()

            return is_last

    def _auto_reset(self) -> None:
        """Auto-reset state when transaction completes."""
        self._transaction_id = None
        self._pending_count = 0
        self._entered_ids.clear()
        self._exited_ids.clear()
        self._abort_flag.clear()
        self._abort_reason = None
        self._failed_tool = None

    def reset(self) -> None:
        """Manually reset all state."""
        with self._state_lock:
            self._auto_reset()
        # Also release lock if held (shouldn't happen normally)
        try:
            self._execution_lock.release()
        except RuntimeError:
            pass  # Lock wasn't held

    def acquire_execution_slot(self, tool_call_id: str) -> "_ExecutionSlotContext":
        """Acquire an execution slot (context manager).

        Args:
            tool_call_id: ID of the tool call

        Returns:
            Context manager that yields ExecutionSlot
        """
        return _ExecutionSlotContext(self, tool_call_id)


class _ExecutionSlotContext:
    """Context manager for sequential execution slot."""

    def __init__(self, lock: SequentialExecutionLock, tool_call_id: str):
        """Initialize context.

        Args:
            lock: The sequential execution lock
            tool_call_id: ID of the tool call
        """
        self._lock = lock
        self._tool_call_id = tool_call_id
        self._slot: ExecutionSlot | None = None
        self._transaction_id: str | None = None

    def __enter__(self) -> ExecutionSlot:
        """Enter the execution slot.

        Tracks entry BEFORE acquiring lock, then checks abort AFTER.
        """
        # Track entry before acquiring lock
        self._transaction_id = self._lock.enter_transaction(self._tool_call_id)

        # Acquire execution lock (blocks)
        self._lock._execution_lock.acquire()

        # Create slot and check for abort
        self._slot = ExecutionSlot(
            tool_call_id=self._tool_call_id,
            transaction_id=self._transaction_id,
        )

        if self._lock.should_abort():
            failed_tool, reason = self._lock.get_abort_info()
            self._slot.should_abort = True
            self._slot.abort_reason = reason

        return self._slot

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the execution slot."""
        # Release lock
        try:
            self._lock._execution_lock.release()
        except RuntimeError:
            pass

        # Track exit
        self._lock.exit_transaction(self._tool_call_id)

        return False  # Don't suppress exceptions
