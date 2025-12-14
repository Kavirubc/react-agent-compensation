"""Batch detection for parallel tool execution.

Detects when multiple tool calls are executing in parallel
by analyzing thread execution patterns within a time window.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field


@dataclass
class CallRecord:
    """Record of a tool call for batch detection."""

    tool_name: str
    thread_id: str
    tool_call_id: str
    timestamp: float


class BatchDetector:
    """Detects parallel tool batches via thread execution patterns.

    Uses a time window to group concurrent tool calls into batches.
    When multiple threads execute tools within the window, they're
    considered part of the same batch.

    Example:
        detector = BatchDetector(time_window_ms=50)

        # First call - no batch yet
        batch_id = detector.record_call("tool1", "thread-1", "tc-1")
        # Returns: None (single execution)

        # Second call within window from different thread
        batch_id = detector.record_call("tool2", "thread-2", "tc-2")
        # Returns: "abc123" (parallel batch detected)
    """

    def __init__(self, time_window_ms: float = 50):
        """Initialize batch detector.

        Args:
            time_window_ms: Time window in milliseconds to detect parallel calls
        """
        self.time_window = time_window_ms / 1000  # Convert to seconds
        self._recent_calls: dict[str, list[CallRecord]] = {}
        self._batch_tool_calls: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def record_call(
        self, tool_name: str, thread_id: str, tool_call_id: str
    ) -> str | None:
        """Record a tool call and detect if part of parallel batch.

        Args:
            tool_name: Name of the tool being called
            thread_id: ID of the thread executing the call
            tool_call_id: Unique ID for this tool call

        Returns:
            batch_id if parallel execution detected, None for single execution
        """
        now = time.time()

        with self._lock:
            # Clean expired entries
            self._clean_expired(now)

            # Add current call
            if tool_name not in self._recent_calls:
                self._recent_calls[tool_name] = []

            record = CallRecord(
                tool_name=tool_name,
                thread_id=thread_id,
                tool_call_id=tool_call_id,
                timestamp=now,
            )
            self._recent_calls[tool_name].append(record)

            # Check for parallel execution (multiple threads in window)
            batch_id = self._detect_batch(tool_name, now)
            if batch_id:
                # Track tool calls in this batch
                if batch_id not in self._batch_tool_calls:
                    self._batch_tool_calls[batch_id] = set()
                self._batch_tool_calls[batch_id].add(tool_call_id)

            return batch_id

    def _clean_expired(self, now: float) -> None:
        """Remove call records outside the time window."""
        cutoff = now - self.time_window

        for tool_name in list(self._recent_calls.keys()):
            self._recent_calls[tool_name] = [
                r for r in self._recent_calls[tool_name] if r.timestamp >= cutoff
            ]
            if not self._recent_calls[tool_name]:
                del self._recent_calls[tool_name]

    def _detect_batch(self, tool_name: str, now: float) -> str | None:
        """Detect if current call is part of a parallel batch.

        Args:
            tool_name: Name of the tool
            now: Current timestamp

        Returns:
            batch_id if parallel detected, None otherwise
        """
        calls = self._recent_calls.get(tool_name, [])
        if len(calls) < 2:
            return None

        cutoff = now - self.time_window
        calls_in_window = [r for r in calls if r.timestamp >= cutoff]

        if len(calls_in_window) < 2:
            return None

        # Check for multiple threads
        threads_in_window = {r.thread_id for r in calls_in_window}
        if len(threads_in_window) < 2:
            return None

        # Generate deterministic batch ID
        sorted_threads = sorted(threads_in_window)
        first_ts = min(r.timestamp for r in calls_in_window)
        batch_id = hashlib.md5(
            f"{tool_name}:{sorted_threads}:{int(first_ts * 1000)}".encode()
        ).hexdigest()[:16]

        return batch_id

    def get_batch_tool_calls(self, batch_id: str) -> set[str]:
        """Get all tool call IDs in a batch.

        Args:
            batch_id: ID of the batch

        Returns:
            Set of tool call IDs
        """
        with self._lock:
            return self._batch_tool_calls.get(batch_id, set()).copy()

    def cleanup_batch(self, batch_id: str) -> None:
        """Clean up tracking for a completed batch.

        Args:
            batch_id: ID of the batch to clean up
        """
        with self._lock:
            self._batch_tool_calls.pop(batch_id, None)

    def reset(self) -> None:
        """Reset all tracking state."""
        with self._lock:
            self._recent_calls.clear()
            self._batch_tool_calls.clear()
