"""
Travel Booking with Automatic Compensation using CrewAI

This example demonstrates a travel booking workflow using CrewAI with
the react-agent-compensation framework. Features:

1. Multi-tool workflow (flight booking, hotel booking)
2. Automatic compensation on failure
3. CompensationSchema for declarative parameter mapping
4. Goal-Aware Recovery guidance

The workflow:
- Book a flight
- Book a hotel
- If any step fails, previous bookings are automatically cancelled

Requirements:
    pip install crewai react-agent-compensation
"""

import logging

# Set up logging to see compensation actions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable compensation logging
comp_logger = logging.getLogger("react_agent_compensation")
comp_logger.setLevel(logging.INFO)


# Simulated booking counter for generating IDs
_booking_counter = {"flight": 0, "hotel": 0}


# Mock tool implementations
def book_flight_impl(destination: str, date: str) -> str:
    """Book a flight to the destination."""
    _booking_counter["flight"] += 1
    booking_id = f"FL-{destination[:3].upper()}-{_booking_counter['flight']:04d}"
    logger.info(f"Booked flight to {destination} on {date}, ID: {booking_id}")
    return f'{{"booking_id": "{booking_id}", "destination": "{destination}", "date": "{date}", "status": "confirmed"}}'


def cancel_flight_impl(booking_id: str) -> str:
    """Cancel a flight booking."""
    logger.info(f"Cancelled flight booking: {booking_id}")
    return f'{{"cancelled": true, "booking_id": "{booking_id}"}}'


def book_hotel_impl(city: str, nights: int, check_in: str = "") -> str:
    """Book a hotel in a city."""
    _booking_counter["hotel"] += 1
    reservation_id = f"HT-{city[:3].upper()}-{_booking_counter['hotel']:04d}"

    # Simulate failure for specific condition
    if nights > 14:
        logger.warning(f"Hotel booking failed: Cannot book more than 14 nights")
        return '{"error": "Cannot book more than 14 nights at once"}'

    logger.info(f"Booked hotel in {city} for {nights} nights, ID: {reservation_id}")
    return f'{{"reservation_id": "{reservation_id}", "city": "{city}", "nights": {nights}, "status": "confirmed"}}'


def cancel_hotel_impl(reservation_id: str) -> str:
    """Cancel a hotel reservation."""
    logger.info(f"Cancelled hotel reservation: {reservation_id}")
    return f'{{"cancelled": true, "reservation_id": "{reservation_id}"}}'


def run_example():
    """Run the CrewAI travel booking example."""
    # Import CrewAI - will fail if not installed
    try:
        from crewai import Agent, Task
        from crewai.tools import tool
    except ImportError:
        print("\nCrewAI is not installed. Install with: pip install crewai")
        print("Showing example code structure instead.\n")
        show_example_structure()
        return

    # Import compensation framework
    from react_agent_compensation.crewai_adaptor import (
        create_compensated_crew,
        RetryPolicy,
    )
    from react_agent_compensation.core.extraction import CompensationSchema

    # Define tools using CrewAI's @tool decorator
    @tool("Book Flight")
    def book_flight(destination: str, date: str) -> str:
        """Book a flight to the destination on the specified date."""
        return book_flight_impl(destination, date)

    @tool("Cancel Flight")
    def cancel_flight(booking_id: str) -> str:
        """Cancel a flight booking using the booking ID."""
        return cancel_flight_impl(booking_id)

    @tool("Book Hotel")
    def book_hotel(city: str, nights: int, check_in: str = "") -> str:
        """Book a hotel in a city for the specified number of nights."""
        return book_hotel_impl(city, nights, check_in)

    @tool("Cancel Hotel")
    def cancel_hotel(reservation_id: str) -> str:
        """Cancel a hotel reservation using the reservation ID."""
        return cancel_hotel_impl(reservation_id)

    # Create travel agent
    travel_agent = Agent(
        role="Travel Agent",
        goal="Book complete travel arrangements for customers",
        backstory="You are an experienced travel agent who books flights and hotels efficiently.",
        tools=[book_flight, cancel_flight, book_hotel, cancel_hotel],
        verbose=True,
    )

    # Define compensation schemas
    compensation_schemas = {
        "Book Flight": CompensationSchema(
            param_mapping={"booking_id": "result.booking_id"}
        ),
        "Book Hotel": CompensationSchema(
            param_mapping={"reservation_id": "result.reservation_id"}
        ),
    }

    # Scenario 1: Successful booking
    print("=" * 70)
    print("Scenario 1: Successful Travel Booking")
    print("=" * 70)

    travel_task = Task(
        description="Book a trip to Paris: a flight on March 15 and a hotel for 5 nights",
        agent=travel_agent,
        expected_output="Complete travel itinerary with confirmation numbers",
    )

    crew = create_compensated_crew(
        agents=[travel_agent],
        tasks=[travel_task],
        compensation_mapping={
            "Book Flight": "Cancel Flight",
            "Book Hotel": "Cancel Hotel",
        },
        compensation_schemas=compensation_schemas,
        retry_policy=RetryPolicy(max_retries=2),
        goals=["minimize_cost", "prefer_direct_flights"],
        verbose=True,
    )

    result = crew.kickoff()
    print(f"\nResult: {result}")

    # Scenario 2: Booking that triggers compensation
    print("\n" + "=" * 70)
    print("Scenario 2: Failed Booking with Automatic Compensation")
    print("=" * 70)
    print("Note: Hotel booking will fail due to >14 nights, triggering rollback")

    travel_task_fail = Task(
        description="Book a trip to Tokyo: a flight on April 1 and a hotel for 20 nights",
        agent=travel_agent,
        expected_output="Complete travel itinerary with confirmation numbers",
    )

    # Create new crew with fresh middleware
    crew_fail = create_compensated_crew(
        agents=[travel_agent],
        tasks=[travel_task_fail],
        compensation_mapping={
            "Book Flight": "Cancel Flight",
            "Book Hotel": "Cancel Hotel",
        },
        compensation_schemas=compensation_schemas,
        retry_policy=RetryPolicy(max_retries=1),
        goals=["minimize_cost", "prefer_direct_flights"],
        verbose=True,
    )

    result = crew_fail.kickoff()
    print(f"\nResult: {result}")

    print("\n" + "=" * 70)
    print("Example completed")
    print("=" * 70)


def show_example_structure():
    """Show the example structure when CrewAI is not installed."""
    print("""
Example Code Structure:

1. Define compensation mapping:
   compensation_mapping = {
       "Book Flight": "Cancel Flight",
       "Book Hotel": "Cancel Hotel",
   }

2. Define parameter extraction schemas:
   compensation_schemas = {
       "Book Flight": CompensationSchema(
           param_mapping={"booking_id": "result.booking_id"}
       ),
   }

3. Create compensated crew:
   crew = create_compensated_crew(
       agents=[travel_agent],
       tasks=[travel_task],
       compensation_mapping=compensation_mapping,
       compensation_schemas=compensation_schemas,
       goals=["minimize_cost", "prefer_direct_flights"],
   )

4. Execute:
   result = crew.kickoff()

When a tool fails:
- Automatic retry attempts with exponential backoff
- Alternative tool fallback if configured
- Rollback of all completed compensatable actions
- Informative message to LLM with Strategic Context Preservation
- Goal-Aware Recovery guidance for replanning
""")


if __name__ == "__main__":
    run_example()
