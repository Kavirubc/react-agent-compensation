"""Thread-safe transaction log for tracking compensatable actions.

This module provides TransactionLog, a thread-safe container for ActionRecords
that supports:
- Adding, updating, and retrieving action records
- Generating rollback plans using topological sort
- Multi-agent filtering and coordination
- Serialization for persistence
"""

from __future__ import annotations

import copy
import threading
from collections import deque
from typing import Any

from react_agent_compensation.core.models import ActionRecord, ActionStatus


class TransactionLog:
    """Thread-safe log for tracking compensatable actions.

    Manages ActionRecords with support for:
    - CRUD operations with thread safety (RLock)
    - Topological sort for correct rollback ordering
    - Multi-agent filtering via agent_id
    - Atomic batch operations
    - Serialization/deserialization

    Example:
        log = TransactionLog()
        record = ActionRecord(action="book_flight", params={"dest": "NYC"})
        log.add(record)
        log.update(record.id, status=ActionStatus.COMPLETED, result={"id": "123"})

        # Get rollback plan (dependents first)
        plan = log.get_rollback_plan()
    """

    def __init__(self, records: dict[str, ActionRecord] | None = None):
        """Initialize the transaction log.

        Args:
            records: Optional initial records dict (for deserialization)
        """
        self._records: dict[str, ActionRecord] = records or {}
        self._lock = threading.RLock()

    def add(self, record: ActionRecord) -> None:
        """Add a new action record to the log.

        Args:
            record: The ActionRecord to add
        """
        with self._lock:
            self._records[record.id] = record

    def update(self, record_id: str, **kwargs: Any) -> None:
        """Update fields on an existing record.

        Args:
            record_id: ID of the record to update
            **kwargs: Fields to update (e.g., status=ActionStatus.COMPLETED)

        Raises:
            KeyError: If record_id not found
        """
        with self._lock:
            if record_id not in self._records:
                raise KeyError(f"Record {record_id} not found")
            record = self._records[record_id]
            for key, value in kwargs.items():
                if hasattr(record, key):
                    setattr(record, key, value)

    def get(self, record_id: str) -> ActionRecord | None:
        """Get a record by ID.

        Args:
            record_id: ID of the record to retrieve

        Returns:
            The ActionRecord or None if not found
        """
        with self._lock:
            return self._records.get(record_id)

    def snapshot(self) -> dict[str, ActionRecord]:
        """Get a deep copy of all records for safe iteration.

        Returns:
            Deep copy of the records dictionary
        """
        with self._lock:
            return copy.deepcopy(self._records)

    def filter_by_agent(self, agent_id: str) -> list[ActionRecord]:
        """Get all records for a specific agent.

        Args:
            agent_id: The agent ID to filter by

        Returns:
            List of records belonging to the agent
        """
        with self._lock:
            return [r for r in self._records.values() if r.agent_id == agent_id]

    def filter_by_status(self, status: ActionStatus) -> list[ActionRecord]:
        """Get all records with a specific status.

        Args:
            status: The status to filter by

        Returns:
            List of records with the given status
        """
        with self._lock:
            return [r for r in self._records.values() if r.status == status]

    def get_rollback_plan(self, agent_id: str | None = None) -> list[ActionRecord]:
        """Get records in correct order for rollback (topological sort).

        Uses Kahn's algorithm to sort records by dependencies, ensuring
        dependents are rolled back before their dependencies.

        Args:
            agent_id: Optional agent ID to filter records

        Returns:
            List of ActionRecords in rollback order (dependents first)
        """
        with self._lock:
            # Filter eligible records: COMPLETED, not compensated, has compensator
            eligible = [
                r for r in self._records.values()
                if r.status == ActionStatus.COMPLETED
                and not r.compensated
                and r.compensator is not None
                and (agent_id is None or r.agent_id == agent_id)
            ]

            if not eligible:
                return []

            # Build ID set for valid records
            eligible_ids = {r.id for r in eligible}

            # Build in-degree map (count of dependents pointing to each node)
            in_degree: dict[str, int] = {r.id: 0 for r in eligible}

            # For each record, increment in-degree of its dependencies
            for record in eligible:
                for dep_id in record.depends_on:
                    if dep_id in in_degree:
                        in_degree[dep_id] += 1

            # Start with nodes that have no dependents (in_degree = 0)
            queue: deque[str] = deque(
                r.id for r in eligible if in_degree[r.id] == 0
            )
            result: list[ActionRecord] = []

            while queue:
                node_id = queue.popleft()
                node = self._records[node_id]
                result.append(node)

                # Decrement in-degree of dependencies
                for dep_id in node.depends_on:
                    if dep_id in in_degree:
                        in_degree[dep_id] -= 1
                        if in_degree[dep_id] == 0:
                            queue.append(dep_id)

            # If we couldn't process all nodes, there's a cycle - fall back to timestamp
            if len(result) < len(eligible):
                # Sort by timestamp descending (most recent first)
                result = sorted(eligible, key=lambda r: r.timestamp, reverse=True)

            return result

    def mark_compensated(self, record_id: str) -> None:
        """Mark a record as compensated.

        Args:
            record_id: ID of the record to mark

        Raises:
            KeyError: If record_id not found
        """
        with self._lock:
            if record_id not in self._records:
                raise KeyError(f"Record {record_id} not found")
            self._records[record_id].mark_compensated()

    def clear(self, agent_id: str | None = None) -> None:
        """Clear records from the log.

        Args:
            agent_id: If provided, only clear records for this agent.
                     If None, clear all records.
        """
        with self._lock:
            if agent_id is None:
                self._records.clear()
            else:
                self._records = {
                    k: v for k, v in self._records.items()
                    if v.agent_id != agent_id
                }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the log to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        with self._lock:
            return {
                "records": {
                    k: v.model_dump() for k, v in self._records.items()
                }
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransactionLog:
        """Deserialize a log from a dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            New TransactionLog instance
        """
        records_data = data.get("records", {})
        records = {
            k: ActionRecord.model_validate(v) for k, v in records_data.items()
        }
        return cls(records=records)

    def __len__(self) -> int:
        """Return the number of records in the log."""
        with self._lock:
            return len(self._records)

    def __contains__(self, record_id: str) -> bool:
        """Check if a record ID exists in the log."""
        with self._lock:
            return record_id in self._records
