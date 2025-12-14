"""Custom error detection strategies for domain-specific responses."""

from examples.strategies.error_detection import PaymentErrorStrategy, BookingErrorStrategy

__all__ = [
    "PaymentErrorStrategy",
    "BookingErrorStrategy",
]
