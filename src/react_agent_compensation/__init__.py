"""react-agent-compensation: Framework-agnostic compensation/rollback for ReAct agents.

This library provides robust execution guarantees for AI agent workflows by
intercepting tool calls to handle retries, alternative paths, and compensation
(rollback) logic when workflows fail.

Quick Start:
    from react_agent_compensation.core import (
        ActionRecord,
        ActionStatus,
        RetryPolicy,
        CompensationPairs,
    )

For LangChain integration:
    from react_agent_compensation.langchain_adaptor import (
        create_compensated_agent,
        CompensationMiddleware,
    )
"""

__version__ = "0.1.0"

# Re-export core components for convenience
from react_agent_compensation.core import (
    ActionExecutor,
    ActionRecord,
    ActionResult,
    ActionStatus,
    AlternativeMap,
    CompensationError,
    CompensationPairs,
    CriticalFailure,
    ExtractionError,
    RecoveryConfig,
    RecoveryError,
    RetryPolicy,
    RollbackFailure,
    SimpleActionResult,
    ToolSchemaProvider,
)

__all__ = [
    "__version__",
    # Models
    "ActionRecord",
    "ActionStatus",
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
    "SimpleActionResult",
]
