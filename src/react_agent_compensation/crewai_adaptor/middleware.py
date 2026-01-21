"""CrewAI middleware adapter for compensation.

Provides CrewAICompensationMiddleware that integrates the Core RecoveryManager
with CrewAI's tool execution pattern.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from react_agent_compensation.core.config import AlternativeMap, CompensationPairs, RetryPolicy
from react_agent_compensation.core.extraction import CompensationSchema, create_extraction_strategy
from react_agent_compensation.core.recovery_manager import RecoveryManager
from react_agent_compensation.core.transaction_log import TransactionLog
from react_agent_compensation.crewai_adaptor.adapters import (
    CrewAIToolExecutor,
    build_tools_cache,
)

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class CrewAICompensationMiddleware:
    """CrewAI middleware for recovery and compensation.

    Coordinates tool calls with:
    - Action recording for compensatable tools
    - Recovery: retry + alternatives on failure
    - Compensation: rollback completed actions on unrecoverable failure

    Example:
        middleware = CrewAICompensationMiddleware(
            compensation_mapping={"book_flight": "cancel_flight"},
            alternative_map={"book_flight": ["book_flight_backup"]},
            tools=tools,
        )

        # Wrap tools with middleware
        wrapped_tools = middleware.wrap_tools(tools)
    """

    def __init__(
        self,
        compensation_mapping: CompensationPairs,
        tools: Any = None,
        *,
        alternative_map: AlternativeMap | None = None,
        retry_policy: RetryPolicy | None = None,
        shared_log: TransactionLog | None = None,
        crew_id: str | None = None,
        agent_id: str | None = None,
        compensation_schemas: dict[str, CompensationSchema] | None = None,
        state_mappers: dict[str, Callable] | None = None,
        use_llm_extraction: bool = False,
        llm_model: str = "gpt-4o-mini",
        goals: list[str] | None = None,
    ):
        """Initialize middleware.

        Args:
            compensation_mapping: Maps tool names to compensation tools
            tools: List of CrewAI tools
            alternative_map: Maps tools to alternatives to try on failure
            retry_policy: Configuration for retry behavior
            shared_log: Shared TransactionLog for multi-agent scenarios
            crew_id: Identifier for the crew (for filtering in multi-crew)
            agent_id: Identifier for this agent in multi-agent scenarios
            compensation_schemas: Declarative extraction schemas
            state_mappers: Custom extraction functions
            use_llm_extraction: Enable LLM-based extraction
            llm_model: Model for LLM extraction
            goals: Optimization goals for goal-aware recovery
        """
        self.compensation_mapping = compensation_mapping
        self.alternative_map = alternative_map or {}
        self.goals = goals or []
        self.crew_id = crew_id
        self.agent_id = agent_id

        self._tools_cache = build_tools_cache(tools)

        # Build extraction strategy
        extraction_strategy = create_extraction_strategy(
            state_mappers=state_mappers,
            compensation_schemas=compensation_schemas,
            include_llm=use_llm_extraction,
            llm_model=llm_model,
        )

        # Create tool executor
        executor = CrewAIToolExecutor(self._tools_cache)

        # Create RecoveryManager
        self._rc_manager = RecoveryManager(
            compensation_pairs=compensation_mapping,
            alternative_map=alternative_map or {},
            retry_policy=retry_policy,
            extraction_strategy=extraction_strategy,
            action_executor=executor,
            agent_id=agent_id,
        )

        # Use shared log if provided
        if shared_log is not None:
            self._rc_manager._log = shared_log

    @property
    def rc_manager(self) -> RecoveryManager:
        """Access to Core's RecoveryManager."""
        return self._rc_manager

    @property
    def transaction_log(self) -> TransactionLog:
        """Access to the transaction log."""
        return self._rc_manager.log

    def is_compensatable(self, tool_name: str) -> bool:
        """Check if a tool has a compensation pair."""
        return self._rc_manager.is_compensatable(tool_name)

    def get_tool(self, name: str) -> Any | None:
        """Get a tool by name from the cache."""
        return self._tools_cache.get(name)

    def rollback(self) -> Any:
        """Manually trigger rollback."""
        return self._rc_manager.rollback()

    def clear(self) -> None:
        """Clear the transaction log."""
        self._rc_manager.clear()

    def add_tool(self, tool: Any) -> None:
        """Add a tool to the cache.

        Args:
            tool: CrewAI tool instance
        """
        if hasattr(tool, "name"):
            self._tools_cache[tool.name] = tool

    def get_failure_summary(self) -> str:
        """Get cumulative failure context summary for LLM."""
        return self._rc_manager.get_failure_summary()
