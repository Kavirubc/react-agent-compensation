"""Interrupt support for human-in-the-loop approval in Strands.

Provides optional interrupt functionality for requiring human approval
before executing compensation actions or high-risk operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from react_agent_compensation.core.models import ActionRecord


logger = logging.getLogger(__name__)


class CompensationApprovalInterrupt:
    """Interrupt handler for compensation approval.

    Use this to require human approval before executing
    compensation/rollback actions.

    Example:
        interrupt = CompensationApprovalInterrupt(
            require_approval=["refund_payment", "release_inventory"],
        )

        provider = CompensationHookProvider(
            compensation_pairs={...},
            approval_interrupt=interrupt,
        )
    """

    def __init__(
        self,
        require_approval: list[str] | None = None,
        approval_callback: Callable[[str, dict[str, Any]], bool] | None = None,
        auto_approve_retry: bool = True,
    ):
        """Initialize interrupt handler.

        Args:
            require_approval: List of tool names requiring approval for compensation
            approval_callback: Custom callback to request approval.
                Receives (action_name, params) and returns True to approve
            auto_approve_retry: Whether to auto-approve retry attempts (vs alternatives)
        """
        self._require_approval = set(require_approval or [])
        self._approval_callback = approval_callback
        self._auto_approve_retry = auto_approve_retry

    def should_interrupt_rollback(
        self,
        records: list["ActionRecord"],
    ) -> bool:
        """Check if rollback should be interrupted for approval.

        Args:
            records: List of ActionRecords that will be compensated

        Returns:
            True if any record's compensator requires approval
        """
        for record in records:
            if record.compensator in self._require_approval:
                return True
        return False

    def should_interrupt_recovery(
        self,
        action: str,
        is_retry: bool,
    ) -> bool:
        """Check if recovery should be interrupted for approval.

        Args:
            action: Name of the action being retried/replaced
            is_retry: True if this is a retry, False if alternative

        Returns:
            True if approval is required
        """
        if self._auto_approve_retry and is_retry:
            return False

        return action in self._require_approval

    async def request_approval(
        self,
        action: str,
        params: dict[str, Any],
        context: str = "",
    ) -> bool:
        """Request approval for an action.

        Args:
            action: Name of the action requiring approval
            params: Parameters for the action
            context: Additional context string

        Returns:
            True if approved, False otherwise
        """
        if self._approval_callback:
            return self._approval_callback(action, params)

        # Default: log and auto-approve (no actual interruption)
        logger.warning(
            f"[INTERRUPT] Approval requested for {action} with params {params}. "
            f"Context: {context}. Auto-approving (no callback configured)."
        )
        return True

    def request_approval_sync(
        self,
        action: str,
        params: dict[str, Any],
        context: str = "",
    ) -> bool:
        """Request approval for an action (sync version).

        Args:
            action: Name of the action requiring approval
            params: Parameters for the action
            context: Additional context string

        Returns:
            True if approved, False otherwise
        """
        if self._approval_callback:
            return self._approval_callback(action, params)

        logger.warning(
            f"[INTERRUPT] Approval requested for {action} with params {params}. "
            f"Context: {context}. Auto-approving (no callback configured)."
        )
        return True


class InteractiveApproval:
    """Interactive console-based approval for development/testing.

    Prompts the user via console for approval decisions.

    Example:
        approval = InteractiveApproval()

        interrupt = CompensationApprovalInterrupt(
            require_approval=["refund_payment"],
            approval_callback=approval.prompt,
        )
    """

    def __init__(self, default_approve: bool = True):
        """Initialize interactive approval.

        Args:
            default_approve: Default response if user just presses Enter
        """
        self._default_approve = default_approve

    def prompt(self, action: str, params: dict[str, Any]) -> bool:
        """Prompt user for approval via console.

        Args:
            action: Name of the action
            params: Parameters for the action

        Returns:
            True if approved, False if denied
        """
        default_str = "Y/n" if self._default_approve else "y/N"
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())

        print(f"\n[APPROVAL REQUIRED]")
        print(f"  Action: {action}")
        print(f"  Params: {param_str}")

        try:
            response = input(f"  Approve? [{default_str}]: ").strip().lower()

            if not response:
                return self._default_approve
            return response in ('y', 'yes', 'true', '1')
        except (EOFError, KeyboardInterrupt):
            print("\n  Approval denied (interrupted)")
            return False


def create_logging_interrupt(
    require_approval: list[str] | None = None,
    log_level: int = logging.INFO,
) -> CompensationApprovalInterrupt:
    """Create an interrupt that logs actions but auto-approves.

    Useful for debugging/auditing without actually blocking.

    Args:
        require_approval: List of tool names to log
        log_level: Logging level to use

    Returns:
        CompensationApprovalInterrupt that logs but auto-approves
    """
    def logging_callback(action: str, params: dict[str, Any]) -> bool:
        logger.log(
            log_level,
            f"[AUDIT] Compensation action: {action} with params {params}"
        )
        return True  # Always approve

    return CompensationApprovalInterrupt(
        require_approval=require_approval,
        approval_callback=logging_callback,
    )


def create_webhook_interrupt(
    require_approval: list[str] | None = None,
    webhook_url: str = "",
    timeout: float = 30.0,
) -> CompensationApprovalInterrupt:
    """Create an interrupt that calls a webhook for approval.

    The webhook receives a POST request with action details and
    should return JSON: {"approved": true/false}

    Args:
        require_approval: List of tool names requiring webhook approval
        webhook_url: URL to POST approval requests to
        timeout: Request timeout in seconds

    Returns:
        CompensationApprovalInterrupt that uses webhook for approval
    """
    import json

    def webhook_callback(action: str, params: dict[str, Any]) -> bool:
        if not webhook_url:
            logger.warning("Webhook URL not configured, auto-approving")
            return True

        try:
            import urllib.request
            import urllib.error

            data = json.dumps({
                "action": action,
                "params": params,
                "type": "compensation_approval",
            }).encode('utf-8')

            req = urllib.request.Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get("approved", False)

        except (urllib.error.URLError, json.JSONDecodeError, Exception) as e:
            logger.error(f"Webhook approval failed: {e}")
            return False

    return CompensationApprovalInterrupt(
        require_approval=require_approval,
        approval_callback=webhook_callback,
    )
