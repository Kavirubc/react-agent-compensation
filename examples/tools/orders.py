"""Order management tools."""

import logging
from typing import Dict, Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def create_order(customer_id: str, items: str) -> Dict[str, Any]:
    """Create a new order for a customer.

    Args:
        customer_id: Customer identifier.
        items: Comma-separated list of product IDs.

    Returns:
        Order details including order_id and status.
    """
    logger.debug("create_order customer_id=%s items=%s", customer_id, items)
    item_count = len(items.split(","))
    order_id = f"ORD-{customer_id}-{item_count}"
    return {
        "order_id": order_id,
        "customer_id": customer_id,
        "items": items,
        "status": "created",
    }


@tool
def cancel_order(order_id: str) -> str:
    """Cancel an existing order.

    Args:
        order_id: Order identifier to cancel.

    Returns:
        Confirmation message.
    """
    logger.debug("cancel_order order_id=%s", order_id)
    return f"Order {order_id} cancelled"
