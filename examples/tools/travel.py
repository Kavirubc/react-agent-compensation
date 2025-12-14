"""Travel booking tools for multi-agent demonstration."""

import logging
from typing import Dict, Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def book_flight(destination: str, departure_date: str) -> Dict[str, Any]:
    """Book a flight to a destination.

    Args:
        destination: Destination city or airport code.
        departure_date: Departure date in YYYY-MM-DD format.

    Returns:
        Flight booking details including flight_id and confirmation.
    """
    logger.debug("book_flight destination=%s date=%s", destination, departure_date)
    flight_id = f"FL-{destination.upper()}-{departure_date.replace('-', '')}"
    return {
        "flight_id": flight_id,
        "destination": destination,
        "departure_date": departure_date,
        "confirmation_code": f"CONF-{flight_id}",
        "status": "confirmed",
    }


@tool
def cancel_flight(flight_id: str) -> str:
    """Cancel a flight booking.

    Args:
        flight_id: Flight booking identifier.

    Returns:
        Confirmation message.
    """
    logger.debug("cancel_flight flight_id=%s", flight_id)
    return f"Flight {flight_id} cancelled"


@tool
def book_hotel(location: str, check_in_date: str, nights: int) -> Dict[str, Any]:
    """Book a hotel at a location.

    Args:
        location: Hotel location (city name).
        check_in_date: Check-in date in YYYY-MM-DD format.
        nights: Number of nights to stay.

    Returns:
        Hotel booking details including hotel_id and confirmation.
        Returns error status for extended stays (7+ nights) due to availability limits.
    """
    logger.debug(
        "book_hotel location=%s check_in=%s nights=%d",
        location,
        check_in_date,
        nights,
    )

    # Simulate unavailability for extended stays (7+ nights)
    if nights >= 7:
        logger.debug("book_hotel failed: no availability for extended stay")
        return {
            "error": f"No availability for {nights}-night stay. Maximum consecutive booking is 6 nights.",
            "status": "error",
            "location": location,
            "nights_requested": nights,
        }

    hotel_id = f"HT-{location.upper().replace(' ', '')}-{check_in_date.replace('-', '')}"
    return {
        "hotel_id": hotel_id,
        "location": location,
        "check_in_date": check_in_date,
        "nights": nights,
        "confirmation_code": f"CONF-{hotel_id}",
        "status": "confirmed",
    }


@tool
def cancel_hotel(hotel_id: str) -> str:
    """Cancel a hotel booking.

    Args:
        hotel_id: Hotel booking identifier.

    Returns:
        Confirmation message.
    """
    logger.debug("cancel_hotel hotel_id=%s", hotel_id)
    return f"Hotel {hotel_id} cancelled"
