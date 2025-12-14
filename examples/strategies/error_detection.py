"""Custom error detection strategies for domain-specific responses."""

import logging
from typing import Optional, Any

from react_agent_compensation.core.errors import ErrorStrategy

logger = logging.getLogger(__name__)


class PaymentErrorStrategy(ErrorStrategy):
    """Detect payment-specific error conditions.

    Checks for payment decline codes, limit violations, and
    transaction failures that require compensation.
    """

    # Known error codes that indicate payment failure
    DECLINE_CODES = {"declined", "insufficient_funds", "expired_card", "invalid_card"}

    def is_error(self, result: Any) -> Optional[bool]:
        """Determine if a payment result indicates an error.

        Args:
            result: Tool result to analyze.

        Returns:
            True if error detected, False if success, None to defer.
        """
        if not hasattr(result, "content"):
            return None

        content = result.content
        if not isinstance(content, dict):
            return None

        # Check for explicit error status
        if content.get("status") == "error":
            logger.debug("PaymentErrorStrategy: detected error status")
            return True

        # Check for error field
        if "error" in content:
            logger.debug("PaymentErrorStrategy: detected error field")
            return True

        # Check for decline codes
        decline_code = content.get("decline_code", "").lower()
        if decline_code in self.DECLINE_CODES:
            logger.debug("PaymentErrorStrategy: detected decline code=%s", decline_code)
            return True

        # Check for transaction failure
        if content.get("transaction_status") == "failed":
            logger.debug("PaymentErrorStrategy: detected transaction failure")
            return True

        # Defer to next strategy if no payment-specific indicators
        return None


class BookingErrorStrategy(ErrorStrategy):
    """Detect booking-specific error conditions.

    Checks for availability issues, reservation conflicts, and
    booking failures in travel and hospitality domains.
    """

    def is_error(self, result: Any) -> Optional[bool]:
        """Determine if a booking result indicates an error.

        Args:
            result: Tool result to analyze.

        Returns:
            True if error detected, False if success, None to defer.
        """
        if not hasattr(result, "content"):
            return None

        content = result.content
        if not isinstance(content, dict):
            return None

        # Check for explicit error status
        if content.get("status") == "error":
            logger.debug("BookingErrorStrategy: detected error status")
            return True

        # Check for availability issues
        if content.get("availability") == "none":
            logger.debug("BookingErrorStrategy: detected no availability")
            return True

        # Check for booking failure
        if content.get("booking_status") in ("failed", "rejected", "cancelled"):
            logger.debug("BookingErrorStrategy: detected booking failure")
            return True

        return None
