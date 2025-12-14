"""Tool definitions for compensation demonstration."""

from examples.tools.inventory import reserve_inventory, release_inventory
from examples.tools.orders import create_order, cancel_order
from examples.tools.payments import process_payment, refund_payment
from examples.tools.shipping import create_shipment, cancel_shipment
from examples.tools.travel import book_flight, cancel_flight, book_hotel, cancel_hotel

__all__ = [
    # Inventory
    "reserve_inventory",
    "release_inventory",
    # Orders
    "create_order",
    "cancel_order",
    # Payments
    "process_payment",
    "refund_payment",
    # Shipping
    "create_shipment",
    "cancel_shipment",
    # Travel
    "book_flight",
    "cancel_flight",
    "book_hotel",
    "cancel_hotel",
]
