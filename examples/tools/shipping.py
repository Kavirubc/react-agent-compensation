"""Shipping and fulfillment tools."""

import logging
from typing import Dict, Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def create_shipment(order_id: str, address: str) -> Dict[str, Any]:
    """Create a shipment for an order.

    Args:
        order_id: Order identifier.
        address: Shipping address.

    Returns:
        Shipment details including shipment_id and tracking_number.
    """
    logger.debug("create_shipment order_id=%s address=%s", order_id, address)
    shipment_id = f"SHIP-{order_id}"
    tracking_number = f"TRACK-{shipment_id}"
    return {
        "shipment_id": shipment_id,
        "order_id": order_id,
        "address": address,
        "tracking_number": tracking_number,
        "status": "created",
    }


@tool
def cancel_shipment(shipment_id: str) -> str:
    """Cancel a pending shipment.

    Args:
        shipment_id: Shipment identifier to cancel.

    Returns:
        Confirmation message.
    """
    logger.debug("cancel_shipment shipment_id=%s", shipment_id)
    return f"Shipment {shipment_id} cancelled"
