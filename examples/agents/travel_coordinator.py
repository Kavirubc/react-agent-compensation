"""Multi-agent travel booking with coordinated compensation."""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver

from react_agent_compensation.core import CompensationSchema
from react_agent_compensation.langchain_adaptor import (
    create_compensated_agent,
    create_multi_agent_log,
)

from examples.config import Config
from examples.tools.travel import book_flight, cancel_flight, book_hotel, cancel_hotel

logger = logging.getLogger(__name__)


@dataclass
class TripRequest:
    """Request to book a complete trip."""

    destination: str
    departure_date: str
    check_in_date: str
    nights: int


@dataclass
class TripResult:
    """Result of trip booking."""

    status: str
    flight_id: str = ""
    hotel_id: str = ""
    error: str = ""
    compensated_agents: List[str] = None

    def __post_init__(self):
        if self.compensated_agents is None:
            self.compensated_agents = []


class TravelCoordinator:
    """Coordinates flight and hotel agents with shared compensation.

    Demonstrates multi-agent compensation where failure in one agent
    triggers rollback of actions from all participating agents.

    Features:
    - Shared TransactionLog across agents
    - MemorySaver checkpointing for fault tolerance
    - Coordinated rollback on any failure
    """

    FLIGHT_AGENT_PROMPT = """You are a flight booking assistant.
Book flights for customers using the available tools.
Provide the flight confirmation details after booking."""

    HOTEL_AGENT_PROMPT = """You are a hotel booking assistant.
Book hotels for customers using the available tools.
Provide the hotel confirmation details after booking."""

    def __init__(self, config: Config):
        """Initialize the travel coordinator.

        Args:
            config: Application configuration.
        """
        self._config = config
        self._shared_log = create_multi_agent_log()
        self._checkpointer = MemorySaver()
        self._flight_agent = self._build_flight_agent()
        self._hotel_agent = self._build_hotel_agent()

    def _build_flight_agent(self):
        """Build the flight booking agent."""
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=self._config.google_api_key,
        )

        compensation_mapping = {"book_flight": "cancel_flight"}

        compensation_schemas = {
            "book_flight": CompensationSchema(
                param_mapping={"flight_id": "result.flight_id"}
            )
        }

        return create_compensated_agent(
            model=model,
            tools=[book_flight, cancel_flight],
            compensation_mapping=compensation_mapping,
            compensation_schemas=compensation_schemas,
            shared_log=self._shared_log,
            agent_id="flight-agent",
            checkpointer=self._checkpointer,
            system_prompt=self.FLIGHT_AGENT_PROMPT,
        )

    def _build_hotel_agent(self):
        """Build the hotel booking agent."""
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=self._config.google_api_key,
        )

        compensation_mapping = {"book_hotel": "cancel_hotel"}

        compensation_schemas = {
            "book_hotel": CompensationSchema(
                param_mapping={"hotel_id": "result.hotel_id"}
            )
        }

        # Note: error_strategies not exposed in new API - using default detection
        return create_compensated_agent(
            model=model,
            tools=[book_hotel, cancel_hotel],
            compensation_mapping=compensation_mapping,
            compensation_schemas=compensation_schemas,
            shared_log=self._shared_log,
            agent_id="hotel-agent",
            checkpointer=self._checkpointer,
            system_prompt=self.HOTEL_AGENT_PROMPT,
        )

    def book_trip_with_hotel_location(
        self,
        request: TripRequest,
        hotel_location: str,
    ) -> TripResult:
        """Book a trip with a specific hotel location override.

        Useful for testing failure scenarios by specifying a location
        that triggers the unavailability check.

        Args:
            request: Trip booking request.
            hotel_location: Override location for hotel booking.

        Returns:
            TripResult with booking details or compensation information.
        """
        return self._book_trip_internal(
            request,
            hotel_location=hotel_location,
        )

    def book_trip(self, request: TripRequest) -> TripResult:
        """Book a complete trip with flight and hotel.

        If any booking fails, all successful bookings are compensated.

        Args:
            request: Trip booking request.

        Returns:
            TripResult with booking details or compensation information.
        """
        return self._book_trip_internal(request, hotel_location=request.destination)

    def _book_trip_internal(
        self,
        request: TripRequest,
        hotel_location: str,
    ) -> TripResult:
        """Internal implementation of trip booking.

        Demonstrates coordinated compensation: if hotel booking fails,
        the previously booked flight is automatically cancelled.

        Args:
            request: Trip booking request.
            hotel_location: Location for hotel booking (can differ from destination).

        Returns:
            TripResult with booking details or compensation information.
        """
        logger.info(
            "trip_booking destination=%s date=%s",
            request.destination,
            request.departure_date,
        )

        # Track successful bookings for potential compensation
        booked_flight_id = None

        # Book flight
        flight_message = (
            f"Book a flight to {request.destination} "
            f"departing on {request.departure_date}."
        )

        flight_config = {
            "run_name": "flight_booking",
            "configurable": {"thread_id": f"trip-{request.destination}"},
            "tags": ["travel", "flight"],
            "metadata": {"destination": request.destination},
        }

        try:
            flight_result = self._flight_agent.invoke(
                {"messages": [("user", flight_message)]},
                config=flight_config,
            )
            booked_flight_id = self._extract_flight_id(flight_result)
            logger.info("flight_booked flight_id=%s", booked_flight_id)
        except Exception as e:
            logger.error("flight_booking error=%s", str(e))
            return TripResult(status="failed", error=f"Flight booking failed: {e}")

        # Book hotel
        hotel_message = (
            f"Book a hotel in {hotel_location} "
            f"checking in on {request.check_in_date} "
            f"for {request.nights} nights."
        )

        hotel_config = {
            "run_name": "hotel_booking",
            "configurable": {"thread_id": f"trip-{request.destination}"},
            "tags": ["travel", "hotel"],
            "metadata": {"location": hotel_location},
        }

        try:
            hotel_result = self._hotel_agent.invoke(
                {"messages": [("user", hotel_message)]},
                config=hotel_config,
            )

            # Check if hotel booking failed
            if self._check_booking_failure(hotel_result):
                logger.info("hotel_booking failed, triggering compensation")
                compensated = self._compensate_flight(booked_flight_id)
                return TripResult(
                    status="compensated",
                    error="Hotel booking failed",
                    compensated_agents=compensated,
                )

            hotel_id = self._extract_hotel_id(hotel_result)
            logger.info("hotel_booked hotel_id=%s", hotel_id)

            return TripResult(
                status="success",
                flight_id=booked_flight_id,
                hotel_id=hotel_id,
            )

        except Exception as e:
            logger.error("hotel_booking error=%s", str(e))
            compensated = self._compensate_flight(booked_flight_id)
            return TripResult(
                status="compensated",
                error=f"Hotel booking failed: {e}",
                compensated_agents=compensated,
            )

    def _compensate_flight(self, flight_id: Optional[str]) -> List[str]:
        """Compensate a booked flight by cancelling it.

        Args:
            flight_id: The flight ID to cancel, or None if no flight was booked.

        Returns:
            List of agent IDs that were compensated.
        """
        if not flight_id:
            return []

        logger.info("compensating flight_id=%s agent=flight-agent", flight_id)

        # Use the flight agent to cancel the booking
        cancel_message = f"Cancel flight {flight_id}."

        try:
            self._flight_agent.invoke(
                {"messages": [("user", cancel_message)]},
                config={
                    "run_name": "flight_cancellation",
                    "configurable": {"thread_id": f"compensation-{flight_id}"},
                    "tags": ["travel", "compensation"],
                },
            )
            logger.info("flight_cancelled flight_id=%s", flight_id)
            return ["flight-agent"]
        except Exception as e:
            logger.error("flight_cancellation error=%s", str(e))
            return []

    def _parse_content(self, content: Any) -> Optional[Dict[str, Any]]:
        """Parse message content to dict."""
        import json

        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    def _check_booking_failure(self, result: Dict[str, Any]) -> bool:
        """Check if a booking result indicates failure."""
        messages = result.get("messages", [])
        for msg in messages:
            if not hasattr(msg, "content"):
                continue

            content = self._parse_content(msg.content)
            if content is None:
                continue

            if content.get("status") == "error":
                return True
            if "error" in content:
                return True

        return False

    def _extract_flight_id(self, result: Dict[str, Any]) -> str:
        """Extract flight ID from agent result."""
        messages = result.get("messages", [])
        for msg in messages:
            if not hasattr(msg, "content"):
                continue

            content = self._parse_content(msg.content)
            if content and "flight_id" in content:
                return content["flight_id"]

        return ""

    def _extract_hotel_id(self, result: Dict[str, Any]) -> str:
        """Extract hotel ID from agent result."""
        messages = result.get("messages", [])
        for msg in messages:
            if not hasattr(msg, "content"):
                continue

            content = self._parse_content(msg.content)
            if content and "hotel_id" in content:
                return content["hotel_id"]

        return ""
