"""LangChain agent example with compensation.

This example demonstrates using the LangChain adaptor to create
a compensated agent using Google's Gemini model.

Prerequisites:
    pip install langchain-google-genai

Set environment variable:
    export GOOGLE_API_KEY=your_api_key
"""

import os
from typing import Any

from langchain_core.tools import tool

from react_agent_compensation.core import CompensationSchema
from react_agent_compensation.langchain_adaptor import (
    CompensationMiddleware,
    create_multi_agent_log,
)


# Define tools with compensation pairs
@tool
def book_flight(destination: str, date: str) -> dict:
    """Book a flight to a destination on a given date."""
    print(f"[TOOL] Booking flight to {destination} on {date}")
    # Simulate booking
    return {
        "booking_id": "FL-2024-001",
        "destination": destination,
        "date": date,
        "status": "confirmed",
    }


@tool
def cancel_flight(booking_id: str, reason: str = "user_request") -> dict:
    """Cancel a flight booking."""
    print(f"[TOOL] Canceling flight {booking_id}: {reason}")
    return {"booking_id": booking_id, "status": "cancelled"}


@tool
def book_hotel(city: str, check_in: str, check_out: str) -> dict:
    """Book a hotel in a city for given dates."""
    print(f"[TOOL] Booking hotel in {city} from {check_in} to {check_out}")
    return {
        "reservation_id": "HT-2024-001",
        "city": city,
        "status": "confirmed",
    }


@tool
def cancel_hotel(reservation_id: str) -> dict:
    """Cancel a hotel reservation."""
    print(f"[TOOL] Canceling hotel {reservation_id}")
    return {"reservation_id": reservation_id, "status": "cancelled"}


def create_gemini_agent():
    """Create a Gemini-based agent with compensation."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        print("Please install: pip install langchain-google-genai")
        return None

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        return None

    # Create the model
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0,
    )

    tools = [book_flight, cancel_flight, book_hotel, cancel_hotel]

    # Create middleware with compensation config
    middleware = CompensationMiddleware(
        compensation_mapping={
            "book_flight": "cancel_flight",
            "book_hotel": "cancel_hotel",
        },
        tools=tools,
        compensation_schemas={
            "book_flight": CompensationSchema(
                param_mapping={"booking_id": "result.booking_id"},
                static_params={"reason": "auto_rollback"},
            ),
            "book_hotel": CompensationSchema(
                param_mapping={"reservation_id": "result.reservation_id"},
            ),
        },
    )

    return model, tools, middleware


def demo_standalone_middleware():
    """Demonstrate middleware without full agent integration."""
    print("\n" + "=" * 60)
    print("Compensation Middleware Demo (Standalone)")
    print("=" * 60)

    # Create tools and middleware
    tools = [book_flight, cancel_flight, book_hotel, cancel_hotel]

    middleware = CompensationMiddleware(
        compensation_mapping={
            "book_flight": "cancel_flight",
            "book_hotel": "cancel_hotel",
        },
        tools=tools,
    )

    print("\n[1] Simulating tool calls with compensation tracking...")

    # Simulate booking a flight
    rec1 = middleware.rc_manager.record_action(
        "book_flight", {"destination": "NYC", "date": "2024-12-20"}
    )
    result1 = book_flight.invoke({"destination": "NYC", "date": "2024-12-20"})
    middleware.rc_manager.mark_completed(rec1.id, result1)
    print(f"    Flight booked: {result1}")

    # Simulate booking a hotel
    rec2 = middleware.rc_manager.record_action(
        "book_hotel", {"city": "NYC", "check_in": "2024-12-20", "check_out": "2024-12-22"}
    )
    result2 = book_hotel.invoke({"city": "NYC", "check_in": "2024-12-20", "check_out": "2024-12-22"})
    middleware.rc_manager.mark_completed(rec2.id, result2)
    print(f"    Hotel booked: {result2}")

    # Show transaction log
    print("\n[2] Transaction Log:")
    for rid, record in middleware.transaction_log.snapshot().items():
        print(f"    {record.action}: {record.status} (compensator: {record.compensator})")

    # Get rollback plan
    print("\n[3] Rollback Plan (if needed):")
    plan = middleware.transaction_log.get_rollback_plan()
    for record in plan:
        print(f"    Would compensate: {record.action} -> {record.compensator}")

    # Demonstrate rollback
    print("\n[4] Simulating failure and triggering rollback...")
    try:
        middleware.rollback()
        print("    Rollback completed successfully!")
    except Exception as e:
        print(f"    Rollback error: {e}")

    # Show final state
    print("\n[5] Final Transaction Log:")
    for rid, record in middleware.transaction_log.snapshot().items():
        print(f"    {record.action}: {record.status}")

    print("\n" + "=" * 60)


def demo_multi_agent():
    """Demonstrate shared log across multiple agents."""
    print("\n" + "=" * 60)
    print("Multi-Agent Shared Log Demo")
    print("=" * 60)

    # Create shared log
    shared_log = create_multi_agent_log()

    # Create middleware for Agent 1 (flight booking)
    agent1_middleware = CompensationMiddleware(
        compensation_mapping={"book_flight": "cancel_flight"},
        tools=[book_flight, cancel_flight],
        shared_log=shared_log,
        agent_id="flight_agent",
    )

    # Create middleware for Agent 2 (hotel booking)
    agent2_middleware = CompensationMiddleware(
        compensation_mapping={"book_hotel": "cancel_hotel"},
        tools=[book_hotel, cancel_hotel],
        shared_log=shared_log,
        agent_id="hotel_agent",
    )

    print("\n[1] Agent 1 books a flight...")
    rec1 = agent1_middleware.rc_manager.record_action(
        "book_flight", {"destination": "LAX", "date": "2024-12-25"}
    )
    result1 = book_flight.invoke({"destination": "LAX", "date": "2024-12-25"})
    agent1_middleware.rc_manager.mark_completed(rec1.id, result1)

    print("\n[2] Agent 2 books a hotel...")
    rec2 = agent2_middleware.rc_manager.record_action(
        "book_hotel", {"city": "LA", "check_in": "2024-12-25", "check_out": "2024-12-27"}
    )
    result2 = book_hotel.invoke({"city": "LA", "check_in": "2024-12-25", "check_out": "2024-12-27"})
    agent2_middleware.rc_manager.mark_completed(rec2.id, result2)

    print("\n[3] Shared Transaction Log (both agents):")
    for rid, record in shared_log.snapshot().items():
        print(f"    [{record.agent_id}] {record.action}: {record.status}")

    print("\n[4] Filtered by agent_id='flight_agent':")
    flight_records = shared_log.filter_by_agent("flight_agent")
    for record in flight_records:
        print(f"    {record.action}: {record.status}")

    print("\n" + "=" * 60)


def main():
    print("React Agent Compensation - LangChain Examples")
    print("=" * 60)

    # Run standalone middleware demo
    demo_standalone_middleware()

    # Run multi-agent demo
    demo_multi_agent()

    # Check if we can run full Gemini agent
    print("\n" + "=" * 60)
    print("Gemini Agent Integration")
    print("=" * 60)

    result = create_gemini_agent()
    if result:
        model, tools, middleware = result
        print("\nGemini agent created successfully!")
        print(f"Model: {model.model}")
        print(f"Tools: {[t.name for t in tools]}")
        print(f"Compensation pairs: {middleware.compensation_mapping}")
    else:
        print("\nSkipping Gemini agent (missing dependencies or API key)")


if __name__ == "__main__":
    main()
