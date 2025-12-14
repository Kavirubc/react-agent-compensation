"""Batch context and intent tracking for parallel tool execution.

This module provides:
- BatchContext: Thread-safe context for parallel tool batches
- IntentNode: Represents a single intended tool call
- IntentDAG: Tracks intended vs actual tool execution
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IntentStatus(str, Enum):
    """Status of an intended tool call."""

    PENDING = "PENDING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


@dataclass
class IntentNode:
    """Represents a single intended tool call in a batch.

    Tracks the lifecycle of a tool call from intent to completion.
    """

    tool_call_id: str
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    status: IntentStatus = IntentStatus.PENDING

    def can_transition_to(self, new_status: IntentStatus) -> bool:
        """Check if transition to new status is valid."""
        valid_transitions = {
            IntentStatus.PENDING: {IntentStatus.EXECUTING, IntentStatus.ABORTED},
            IntentStatus.EXECUTING: {
                IntentStatus.COMPLETED,
                IntentStatus.FAILED,
                IntentStatus.ABORTED,
            },
        }
        return new_status in valid_transitions.get(self.status, set())


@dataclass
class BatchContext:
    """Thread-safe context for parallel tool call batches.

    Tracks batch state and provides lock-free abort signaling
    for fail-fast behavior in parallel execution.
    """

    batch_id: str
    tool_count: int
    tool_call_ids: list[str] = field(default_factory=list)
    abort_flag: threading.Event = field(default_factory=threading.Event)
    abort_reason: str | None = None
    failed_tool: str | None = None
    failed_tool_call_id: str | None = None
    _executed_count: int = 0
    _completed_ids: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def signal_abort(self, tool_name: str, tool_call_id: str, reason: str) -> bool:
        """Signal that the batch should abort.

        Args:
            tool_name: Name of the tool that failed
            tool_call_id: ID of the failed tool call
            reason: Reason for abort

        Returns:
            True if this was the first abort signal, False if already aborted
        """
        if self.abort_flag.is_set():
            return False

        with self._lock:
            if self.abort_flag.is_set():
                return False
            self.abort_reason = reason
            self.failed_tool = tool_name
            self.failed_tool_call_id = tool_call_id
            self.abort_flag.set()
            return True

    def should_abort(self) -> bool:
        """Check if batch should abort (lock-free read)."""
        return self.abort_flag.is_set()

    def record_execution(self, tool_call_id: str | None = None) -> int:
        """Record that a tool has executed.

        Args:
            tool_call_id: Optional ID of the executed tool call

        Returns:
            Current count of executed tools
        """
        with self._lock:
            self._executed_count += 1
            if tool_call_id:
                self._completed_ids.add(tool_call_id)
            return self._executed_count

    def is_complete(self) -> bool:
        """Check if all tools in batch have executed."""
        with self._lock:
            return self._executed_count >= self.tool_count

    def get_orphan_executions(self) -> list[str]:
        """Get tool call IDs that executed after abort signal.

        These are tools that ran even though abort was signaled,
        and may need compensation.

        Returns:
            List of tool call IDs that executed after abort
        """
        with self._lock:
            if self.failed_tool_call_id:
                return [
                    tc_id
                    for tc_id in self._completed_ids
                    if tc_id != self.failed_tool_call_id
                ]
            return list(self._completed_ids)


class IntentDAG:
    """Tracks LLM's intended tool calls vs actual execution.

    Useful for understanding what the LLM intended to execute
    versus what actually ran, especially when abort occurs.
    """

    def __init__(self, batch_id: str):
        """Initialize intent DAG.

        Args:
            batch_id: ID of the batch being tracked
        """
        self.batch_id = batch_id
        self.nodes: dict[str, IntentNode] = {}
        self._lock = threading.Lock()

    def add_intent(
        self, tool_call_id: str, tool_name: str, args: dict[str, Any]
    ) -> IntentNode:
        """Add an intended tool call to the DAG.

        Args:
            tool_call_id: Unique ID for this tool call
            tool_name: Name of the tool
            args: Arguments for the tool

        Returns:
            The created IntentNode
        """
        with self._lock:
            node = IntentNode(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                args=args,
            )
            self.nodes[tool_call_id] = node
            return node

    def mark_executing(self, tool_call_id: str) -> bool:
        """Mark a tool call as executing.

        Returns:
            True if transition was valid, False otherwise
        """
        return self._transition(tool_call_id, IntentStatus.EXECUTING)

    def mark_completed(self, tool_call_id: str) -> bool:
        """Mark a tool call as completed."""
        return self._transition(tool_call_id, IntentStatus.COMPLETED)

    def mark_failed(self, tool_call_id: str) -> bool:
        """Mark a tool call as failed."""
        return self._transition(tool_call_id, IntentStatus.FAILED)

    def mark_aborted(self, tool_call_id: str) -> bool:
        """Mark a tool call as aborted."""
        return self._transition(tool_call_id, IntentStatus.ABORTED)

    def _transition(self, tool_call_id: str, new_status: IntentStatus) -> bool:
        """Perform status transition if valid."""
        with self._lock:
            node = self.nodes.get(tool_call_id)
            if node and node.can_transition_to(new_status):
                node.status = new_status
                return True
            return False

    def abort_pending(self) -> list[str]:
        """Mark all pending nodes as aborted.

        Returns:
            List of tool call IDs that were aborted
        """
        aborted = []
        with self._lock:
            for tool_call_id, node in self.nodes.items():
                if node.status == IntentStatus.PENDING:
                    node.status = IntentStatus.ABORTED
                    aborted.append(tool_call_id)
        return aborted

    def get_report(self) -> dict[str, Any]:
        """Get status report of all intents.

        Returns:
            Dict with counts and categorized tool call IDs
        """
        with self._lock:
            report = {
                "batch_id": self.batch_id,
                "total_tools": len(self.nodes),
                "status_counts": {},
                "completed": [],
                "failed": [],
                "aborted": [],
                "pending": [],
                "executing": [],
            }

            for status in IntentStatus:
                report["status_counts"][status.value] = 0

            for tool_call_id, node in self.nodes.items():
                report["status_counts"][node.status.value] += 1
                status_key = node.status.value.lower()
                if status_key in report:
                    report[status_key].append(tool_call_id)

            return report
