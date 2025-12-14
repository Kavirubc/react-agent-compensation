"""Payment processing tools."""

import logging
from typing import Dict, Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Payment limit for demonstration purposes
PAYMENT_LIMIT = 10000.0


@tool
def process_payment(order_id: str, amount: float, payment_method: str) -> Dict[str, Any]:
    """Process payment for an order.

    Args:
        order_id: Order identifier.
        amount: Payment amount in dollars.
        payment_method: Payment method (e.g., credit_card, paypal).

    Returns:
        Payment details including payment_id and status.
        Returns error status if amount exceeds limit.
    """
    logger.debug(
        "process_payment order_id=%s amount=%.2f method=%s",
        order_id,
        amount,
        payment_method,
    )

    if amount > PAYMENT_LIMIT:
        logger.debug("process_payment failed: amount exceeds limit")
        return {
            "error": f"Payment amount exceeds limit of ${PAYMENT_LIMIT:.2f}",
            "status": "error",
            "order_id": order_id,
        }

    payment_id = f"PAY-{order_id}-{int(amount)}"
    transaction_id = f"TXN-{payment_id}"
    return {
        "payment_id": payment_id,
        "transaction_id": transaction_id,
        "order_id": order_id,
        "amount": amount,
        "payment_method": payment_method,
        "status": "processed",
    }


@tool
def refund_payment(payment_id: str, transaction_id: str) -> str:
    """Refund a processed payment.

    Args:
        payment_id: Payment identifier.
        transaction_id: Transaction identifier.

    Returns:
        Confirmation message.
    """
    logger.debug(
        "refund_payment payment_id=%s transaction_id=%s",
        payment_id,
        transaction_id,
    )
    return f"Payment {payment_id} (transaction {transaction_id}) refunded"
