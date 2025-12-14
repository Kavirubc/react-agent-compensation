"""Recovery Manager - the brain of the compensation system.

Orchestrates retry, alternatives, and rollback logic for compensatable
tool executions.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from react_agent_compensation.core.config import AlternativeMap, CompensationPairs, RetryPolicy
from react_agent_compensation.core.exceptions import CriticalFailure, ExtractionError, RollbackFailure
from react_agent_compensation.core.extraction import create_extraction_strategy
from react_agent_compensation.core.extraction.base import ExtractionStrategy
from react_agent_compensation.core.extraction.path_resolver import extract_all_values
from react_agent_compensation.core.models import ActionRecord, ActionStatus
from react_agent_compensation.core.protocols import ActionExecutor
from react_agent_compensation.core.retry import ExponentialBackoffStrategy, RetryContext, RetryStrategy
from react_agent_compensation.core.transaction_log import TransactionLog

if TYPE_CHECKING:
    from react_agent_compensation.core.errors.base import ErrorStrategy


logger = logging.getLogger(__name__)


class RecoveryResult(BaseModel):
    """Result of a recovery attempt (retry/alternatives)."""

    success: bool
    action_taken: str = ""  # "retry", "alternative", "failed"
    result: Any = None
    attempts: int = 0
    error: str | None = None


class RollbackResult(BaseModel):
    """Result of a rollback operation."""

    success: bool
    compensated: list[str] = Field(default_factory=list)  # Record IDs
    failed: list[str] = Field(default_factory=list)  # Record IDs
    message: str = ""

    model_config = {"arbitrary_types_allowed": True}


class RecoveryManager:
    """Orchestrates retry, alternatives, and rollback logic.

    The "brain" of the compensation system. Handles:
    - Recording actions before execution
    - Marking actions complete/failed
    - Retry logic with configurable strategies
    - Alternative action fallback
    - Rollback with proper dependency ordering

    Example:
        manager = RecoveryManager(
            compensation_pairs={"book_flight": "cancel_flight"},
            alternative_map={"book_flight": ["book_flight_backup"]},
            retry_policy=RetryPolicy(max_retries=3),
        )

        # Record before execution
        record = manager.record_action("book_flight", {"dest": "NYC"})

        # Mark complete on success
        manager.mark_completed(record.id, result={"booking_id": "123"})

        # Or on failure, try recovery
        recovery = manager.recover(record.id, error)
        if not recovery.success:
            rollback = manager.rollback()
    """

    def __init__(
        self,
        compensation_pairs: CompensationPairs,
        alternative_map: AlternativeMap | None = None,
        retry_policy: RetryPolicy | None = None,
        retry_strategy: RetryStrategy | None = None,
        extraction_strategy: ExtractionStrategy | None = None,
        error_strategy: "ErrorStrategy | None" = None,
        action_executor: ActionExecutor | None = None,
        agent_id: str | None = None,
        infer_dependencies: bool = True,
    ):
        """Initialize recovery manager.

        Args:
            compensation_pairs: Maps tool names to compensation tools
            alternative_map: Maps tools to alternative tools to try
            retry_policy: Configuration for retry behavior
            retry_strategy: Strategy for retry decisions (overrides policy)
            extraction_strategy: Strategy for extracting compensation params
            error_strategy: Strategy for detecting errors in results
            action_executor: Executor for running tools
            agent_id: Identifier for multi-agent scenarios
            infer_dependencies: Auto-detect dependencies via data flow
        """
        self._compensation_pairs = compensation_pairs
        self._alternative_map = alternative_map or {}
        self._retry_policy = retry_policy or RetryPolicy()
        self._retry_strategy = retry_strategy or ExponentialBackoffStrategy(self._retry_policy)
        self._extraction_strategy = extraction_strategy or create_extraction_strategy()
        self._error_strategy = error_strategy
        self._action_executor = action_executor
        self._agent_id = agent_id
        self._infer_dependencies = infer_dependencies
        self._log = TransactionLog()

    @property
    def log(self) -> TransactionLog:
        """Access the transaction log."""
        return self._log

    @property
    def compensation_pairs(self) -> CompensationPairs:
        """Get compensation pair mappings."""
        return self._compensation_pairs

    def is_compensatable(self, action: str) -> bool:
        """Check if an action has a compensation pair."""
        return action in self._compensation_pairs

    def record_action(self, action: str, params: dict[str, Any]) -> ActionRecord:
        """Record an action before execution.

        Args:
            action: Name of the tool/action
            params: Parameters passed to the action

        Returns:
            Created ActionRecord with PENDING status
        """
        compensator = self._compensation_pairs.get(action)
        depends_on = self._infer_action_dependencies(params) if self._infer_dependencies else []

        record = ActionRecord(
            action=action,
            params=params,
            status=ActionStatus.PENDING,
            compensator=compensator,
            depends_on=depends_on,
            agent_id=self._agent_id,
        )
        self._log.add(record)
        logger.debug(f"Recorded action {action} with ID {record.id}")
        return record

    def mark_completed(self, record_id: str, result: Any) -> None:
        """Mark an action as successfully completed.

        Args:
            record_id: ID of the action record
            result: Result returned by the action
        """
        self._log.update(record_id, status=ActionStatus.COMPLETED, result=result)
        logger.debug(f"Marked action {record_id} as COMPLETED")

    def mark_failed(self, record_id: str, error: str | None = None) -> None:
        """Mark an action as failed.

        Args:
            record_id: ID of the action record
            error: Error message
        """
        result = {"error": error} if error else None
        self._log.update(record_id, status=ActionStatus.FAILED, result=result)
        logger.debug(f"Marked action {record_id} as FAILED: {error}")

    def _infer_action_dependencies(self, params: dict[str, Any]) -> list[str]:
        """Infer dependencies by matching params to previous results."""
        param_values = extract_all_values(params)
        if not param_values:
            return []

        dependencies = []
        for record_id, record in self._log.snapshot().items():
            if record.status != ActionStatus.COMPLETED:
                continue
            if record.result is None:
                continue

            result_values = extract_all_values(record.result)
            if param_values & result_values:  # Set intersection
                dependencies.append(record_id)

        return dependencies

    def recover(
        self,
        record_id: str,
        error: Exception | str,
        execute_fn: Any | None = None,
    ) -> RecoveryResult:
        """Attempt recovery via retry or alternatives.

        Args:
            record_id: ID of the failed action record
            error: The error that occurred
            execute_fn: Optional function to execute actions

        Returns:
            RecoveryResult indicating success/failure
        """
        record = self._log.get(record_id)
        if not record:
            return RecoveryResult(success=False, error=f"Record {record_id} not found")

        # Try retries first
        context = RetryContext(
            error=error if isinstance(error, Exception) else Exception(error),
            attempt=1,
            action=record.action,
            params=record.params,
        )

        while self._retry_strategy.should_retry(context):
            delay = self._retry_strategy.get_delay(context)
            logger.info(f"Retrying {record.action} after {delay:.2f}s (attempt {context.attempt})")
            time.sleep(delay)

            try:
                result = self._execute_action(record.action, record.params, execute_fn)
                self.mark_completed(record_id, result)
                return RecoveryResult(
                    success=True,
                    action_taken="retry",
                    result=result,
                    attempts=context.attempt,
                )
            except Exception as e:
                context.error = e
                context.attempt += 1

        # Try alternatives
        alternatives = self._alternative_map.get(record.action, [])
        for alt_action in alternatives:
            logger.info(f"Trying alternative: {alt_action}")
            try:
                result = self._execute_action(alt_action, record.params, execute_fn)
                # Update record with alternative action
                self._log.update(record_id, action=alt_action, status=ActionStatus.COMPLETED, result=result)
                return RecoveryResult(
                    success=True,
                    action_taken="alternative",
                    result=result,
                    attempts=context.attempt,
                )
            except Exception:
                continue

        return RecoveryResult(
            success=False,
            action_taken="failed",
            attempts=context.attempt,
            error=str(error),
        )

    def _execute_action(
        self, action: str, params: dict[str, Any], execute_fn: Any | None
    ) -> Any:
        """Execute an action using provided executor or function."""
        if execute_fn:
            return execute_fn(action, params)
        if self._action_executor:
            return self._action_executor.execute(action, params)
        raise RuntimeError("No executor available")

    def rollback(self, failed_record_id: str | None = None) -> RollbackResult:
        """Execute rollback for completed actions.

        Args:
            failed_record_id: Optional ID of the record that triggered rollback

        Returns:
            RollbackResult with compensation details
        """
        plan = self._log.get_rollback_plan(agent_id=self._agent_id)
        if not plan:
            return RollbackResult(success=True, message="No actions to rollback")

        compensated: list[str] = []
        failed: list[str] = []

        for record in plan:
            if not record.compensator:
                continue

            try:
                comp_params = self._extract_compensation_params(record)
                self._execute_compensation(record.compensator, comp_params)
                self._log.mark_compensated(record.id)
                compensated.append(record.id)
                logger.info(f"Compensated {record.action} with {record.compensator}")
            except Exception as e:
                failed.append(record.id)
                logger.error(f"Compensation failed for {record.action}: {e}")
                if len(failed) > 0 and len(compensated) == 0:
                    # First compensation failed - critical
                    raise CriticalFailure(
                        f"Compensation failed immediately: {e}",
                        context={"record_id": record.id, "action": record.action},
                    ) from e

        success = len(failed) == 0
        if not success:
            raise RollbackFailure(
                f"Partial rollback: {len(compensated)} succeeded, {len(failed)} failed",
                failed_records=failed,
                compensated_records=compensated,
            )

        return RollbackResult(
            success=True,
            compensated=compensated,
            failed=failed,
            message=f"Rolled back {len(compensated)} actions",
        )

    def _extract_compensation_params(self, record: ActionRecord) -> dict[str, Any]:
        """Extract parameters for compensation tool."""
        try:
            return self._extraction_strategy.extract(
                result=record.result,
                original_params=record.params,
                tool_name=record.action,
            ) or {}
        except Exception as e:
            raise ExtractionError(
                f"Failed to extract compensation params for {record.action}",
                tool_name=record.action,
                result=record.result,
            ) from e

    def _execute_compensation(self, action: str, params: dict[str, Any]) -> Any:
        """Execute a compensation action."""
        if self._action_executor:
            return self._action_executor.execute(action, params)
        raise RuntimeError("No executor available for compensation")

    def clear(self) -> None:
        """Clear the transaction log."""
        self._log.clear(agent_id=self._agent_id)

    def add_compensation_pair(self, forward: str, compensator: str) -> None:
        """Add a compensation pair at runtime.

        Args:
            forward: Name of the forward action
            compensator: Name of the compensation action
        """
        self._compensation_pairs[forward] = compensator

    def remove_compensation_pair(self, forward: str) -> bool:
        """Remove a compensation pair.

        Args:
            forward: Name of the forward action

        Returns:
            True if removed, False if not found
        """
        if forward in self._compensation_pairs:
            del self._compensation_pairs[forward]
            return True
        return False
