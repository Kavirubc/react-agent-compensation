"""Batch/parallel execution control for compensation operations.

This module provides components for handling parallel tool execution:
- BatchContext: Thread-safe context for parallel batches
- IntentNode, IntentDAG: Track intended vs actual execution
- BatchDetector: Detect parallel execution patterns
- SequentialExecutionLock: Force sequential execution
- BatchManager: Orchestrate all batch components
"""

from react_agent_compensation.core.batch.context import (
    BatchContext,
    IntentDAG,
    IntentNode,
    IntentStatus,
)
from react_agent_compensation.core.batch.detector import BatchDetector, CallRecord
from react_agent_compensation.core.batch.lock import ExecutionSlot, SequentialExecutionLock
from react_agent_compensation.core.batch.manager import BatchManager

__all__ = [
    # Context
    "BatchContext",
    "IntentNode",
    "IntentDAG",
    "IntentStatus",
    # Detector
    "BatchDetector",
    "CallRecord",
    # Lock
    "SequentialExecutionLock",
    "ExecutionSlot",
    # Manager
    "BatchManager",
]
