"""Core module for react-agent-compensation.

This module contains framework-agnostic logic for compensation and recovery.
It has no dependencies on LangChain or other agent frameworks.

Main components:
- ActionRecord, ActionStatus: Data models for tracking actions
- TransactionLog: Thread-safe log for recording actions
- RecoveryManager: Orchestrates retry, alternatives, and rollback
- RetryPolicy, CompensationPairs: Configuration types
- Protocols: Abstract interfaces for framework integration
"""

from react_agent_compensation.core.config import (
    AlternativeMap,
    CompensationPairs,
    RecoveryConfig,
    RetryPolicy,
)
from react_agent_compensation.core.exceptions import (
    CompensationError,
    CriticalFailure,
    ExtractionError,
    RecoveryError,
    RollbackFailure,
)
from react_agent_compensation.core.models import ActionRecord, ActionStatus
from react_agent_compensation.core.transaction_log import TransactionLog
from react_agent_compensation.core.protocols import (
    ActionExecutor,
    ActionResult,
    LLMProvider,
    SimpleActionResult,
    ToolSchemaProvider,
)
from react_agent_compensation.core.recovery_manager import (
    RecoveryManager,
    RecoveryResult,
    RollbackResult,
)
from react_agent_compensation.core.extraction import (
    CompensationSchema,
    create_extraction_strategy,
)

__all__ = [
    # Models
    "ActionRecord",
    "ActionStatus",
    "TransactionLog",
    # Config
    "RetryPolicy",
    "CompensationPairs",
    "AlternativeMap",
    "RecoveryConfig",
    # Exceptions
    "CompensationError",
    "RollbackFailure",
    "CriticalFailure",
    "ExtractionError",
    "RecoveryError",
    # Protocols
    "ActionResult",
    "ActionExecutor",
    "ToolSchemaProvider",
    "LLMProvider",
    "SimpleActionResult",
    # Recovery
    "RecoveryManager",
    "RecoveryResult",
    "RollbackResult",
    # Extraction
    "CompensationSchema",
    "create_extraction_strategy",
]
