"""Inventory management tools."""

import logging
from typing import Dict, Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def reserve_inventory(product_id: str, quantity: int) -> Dict[str, Any]:
    """Reserve inventory for a product.

    Args:
        product_id: Product identifier.
        quantity: Number of units to reserve.

    Returns:
        Reservation details including reservation_id and status.
    """
    logger.debug("reserve_inventory product_id=%s quantity=%d", product_id, quantity)
    reservation_id = f"RES-{product_id}-{quantity}"
    return {
        "reservation_id": reservation_id,
        "product_id": product_id,
        "quantity": quantity,
        "status": "reserved",
    }


@tool
def release_inventory(reservation_id: str) -> str:
    """Release a previously reserved inventory.

    Args:
        reservation_id: Reservation identifier to release.

    Returns:
        Confirmation message.
    """
    logger.debug("release_inventory reservation_id=%s", reservation_id)
    return f"Inventory reservation {reservation_id} released"
