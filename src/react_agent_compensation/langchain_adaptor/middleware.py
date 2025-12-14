"""LangChain middleware adapter for compensation.

Provides CompensationMiddleware that integrates the Core RecoveryManager
with LangChain's AgentMiddleware pattern.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from react_agent_compensation.core.config import AlternativeMap, CompensationPairs, RetryPolicy
from react_agent_compensation.core.extraction import CompensationSchema, create_extraction_strategy
from react_agent_compensation.core.recovery_manager import RecoveryManager
from react_agent_compensation.core.transaction_log import TransactionLog
from react_agent_compensation.langchain_adaptor.adapters import (
    LangChainToolExecutor,
    build_tools_cache,
)
from react_agent_compensation.langchain_adaptor.interceptors import ToolCallInterceptor
from react_agent_compensation.langchain_adaptor.state import (
    ACTION_LOG_KEY,
    get_action_log,
    sync_action_log,
)

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class CompensationMiddleware:
    """LangChain middleware for recovery and compensation.

    Wraps tool calls with:
    - Action recording for compensatable tools
    - Recovery: retry + alternatives on failure
    - Compensation: rollback completed actions on unrecoverable failure

    Example:
        middleware = CompensationMiddleware(
            compensation_mapping={"book_flight": "cancel_flight"},
            alternative_map={"book_flight": ["book_flight_backup"]},
            tools=tools,
        )

        # Use with LangGraph agent
        agent = create_agent(model, tools=tools, middleware=[middleware])
    """

    def __init__(
        self,
        compensation_mapping: CompensationPairs,
        tools: Any = None,
        *,
        alternative_map: AlternativeMap | None = None,
        retry_policy: RetryPolicy | None = None,
        shared_log: TransactionLog | None = None,
        agent_id: str | None = None,
        compensation_schemas: dict[str, CompensationSchema] | None = None,
        state_mappers: dict[str, Callable] | None = None,
        use_llm_extraction: bool = False,
        llm_model: str = "gpt-4o-mini",
    ):
        """Initialize middleware.

        Args:
            compensation_mapping: Maps tool names to compensation tools
            tools: List of LangChain tools
            alternative_map: Maps tools to alternatives to try on failure
            retry_policy: Configuration for retry behavior
            shared_log: Shared TransactionLog for multi-agent scenarios
            agent_id: Identifier for this agent in multi-agent scenarios
            compensation_schemas: Declarative extraction schemas
            state_mappers: Custom extraction functions
            use_llm_extraction: Enable LLM-based extraction
            llm_model: Model for LLM extraction
        """
        self.compensation_mapping = compensation_mapping
        self.alternative_map = alternative_map or {}

        self._tools_cache = build_tools_cache(tools)

        # Build extraction strategy
        extraction_strategy = create_extraction_strategy(
            state_mappers=state_mappers,
            compensation_schemas=compensation_schemas,
            include_llm=use_llm_extraction,
            llm_model=llm_model,
        )

        # Create tool executor
        executor = LangChainToolExecutor(self._tools_cache)

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

        # Create interceptor
        self._interceptor = ToolCallInterceptor(
            rc_manager=self._rc_manager,
            tools_cache=self._tools_cache,
        )

    @property
    def rc_manager(self) -> RecoveryManager:
        """Access to Core's RecoveryManager."""
        return self._rc_manager

    @property
    def transaction_log(self) -> TransactionLog:
        """Access to the transaction log."""
        return self._rc_manager.log

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """Main middleware hook for LangChain.

        This method is called by LangChain's middleware system
        for each tool call.

        Args:
            request: LangChain's ToolCallRequest
            handler: Next handler in chain

        Returns:
            ToolMessage or Command from execution
        """
        # Extract call info from request
        tool_call = getattr(request, "tool_call", {})
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name", "")
            params = tool_call.get("args", {})
            tool_call_id = tool_call.get("id", "")
        else:
            tool_name = getattr(tool_call, "name", "")
            params = getattr(tool_call, "args", {})
            tool_call_id = getattr(tool_call, "id", "")

        # Sync log from state if available
        state = getattr(request, "state", {})
        if state:
            existing_log = get_action_log(state)
            if existing_log:
                self._rc_manager._log = existing_log

        # Execute with interception
        def execute_fn() -> Any:
            return handler(request)

        result = self._interceptor.intercept(
            tool_name=tool_name,
            params=params,
            tool_call_id=tool_call_id,
            execute_fn=execute_fn,
        )

        # Sync log back to state
        if state:
            sync_action_log(state, self._rc_manager.log)

        # Convert to ToolMessage
        return result.to_tool_message(tool_call_id, tool_name)

    def rollback(self) -> None:
        """Manually trigger rollback."""
        self._rc_manager.rollback()

    def clear(self) -> None:
        """Clear the transaction log."""
        self._rc_manager.clear()

    def add_tool(self, tool: Any) -> None:
        """Add a tool to the cache.

        Args:
            tool: LangChain tool instance
        """
        if hasattr(tool, "name"):
            self._tools_cache[tool.name] = tool
