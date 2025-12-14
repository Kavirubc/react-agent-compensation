"""React Agent Compensation demonstration scenarios.

This module demonstrates the react-agent-compensation middleware with:
1. Single-agent order processing with automatic rollback
2. Multi-agent travel booking with coordinated compensation
3. Custom error detection strategies
4. Checkpointing for fault tolerance

Run with: python -m examples.main
"""

import logging
import sys

from langsmith import traceable

from examples.config import load_config, configure_logging
from examples.agents import OrderProcessorAgent, OrderRequest, TravelCoordinator, TripRequest

logger = logging.getLogger("compensation_demo")


@traceable(name="scenario_successful_order", run_type="chain")
def run_successful_order(agent: OrderProcessorAgent) -> None:
    """Demonstrate successful order processing."""
    logger.info("scenario=successful_order status=starting")

    request = OrderRequest(
        customer_id="CUST001",
        items="PROD1,PROD2,PROD3",
        amount=150.00,
        payment_method="credit_card",
        shipping_address="123 Main St, City, State 12345",
    )

    result = agent.process_order(request)

    if result.success:
        logger.info(
            "scenario=successful_order status=completed order_id=%s",
            result.order_id,
        )
    else:
        logger.info(
            "scenario=successful_order status=failed error=%s",
            result.error,
        )


@traceable(name="scenario_payment_failure", run_type="chain")
def run_failed_payment_order(agent: OrderProcessorAgent) -> None:
    """Demonstrate payment failure with automatic compensation."""
    logger.info("scenario=payment_failure status=starting")

    request = OrderRequest(
        customer_id="CUST002",
        items="PROD4,PROD5",
        amount=15000.00,  # Exceeds payment limit
        payment_method="credit_card",
        shipping_address="456 Oak Ave, City, State 67890",
    )

    result = agent.process_order(request)

    if result.compensated_count > 0:
        logger.info(
            "scenario=payment_failure status=compensated actions=%d",
            result.compensated_count,
        )
    else:
        logger.info(
            "scenario=payment_failure status=failed error=%s",
            result.error,
        )


@traceable(name="scenario_successful_trip", run_type="chain")
def run_successful_trip(coordinator: TravelCoordinator) -> None:
    """Demonstrate successful multi-agent trip booking."""
    logger.info("scenario=successful_trip status=starting")

    request = TripRequest(
        destination="Paris",
        departure_date="2024-06-15",
        check_in_date="2024-06-15",
        nights=5,
    )

    result = coordinator.book_trip(request)

    if result.status == "success":
        logger.info(
            "scenario=successful_trip status=completed flight=%s hotel=%s",
            result.flight_id,
            result.hotel_id,
        )
    else:
        logger.info(
            "scenario=successful_trip status=%s error=%s",
            result.status,
            result.error,
        )


@traceable(name="scenario_multi_agent_compensation", run_type="chain")
def run_failed_hotel_booking(coordinator: TravelCoordinator) -> None:
    """Demonstrate hotel failure with coordinated multi-agent compensation."""
    logger.info("scenario=multi_agent_failure status=starting")

    # Extended stay (7+ nights) triggers hotel unavailability
    # This causes the hotel booking to fail after flight is already booked
    request = TripRequest(
        destination="Tokyo",
        departure_date="2024-07-20",
        check_in_date="2024-07-20",
        nights=10,  # Extended stay triggers failure
    )

    result = coordinator.book_trip(request)

    if result.status == "compensated":
        logger.info(
            "scenario=multi_agent_failure status=compensated agents=%d",
            len(result.compensated_agents),
        )
    else:
        logger.info(
            "scenario=multi_agent_failure status=%s error=%s",
            result.status,
            result.error if result.error else "none",
        )


def run_order_scenarios(config) -> None:
    """Run all order processing scenarios."""
    logger.info("section=order_processing")

    agent = OrderProcessorAgent(config)

    run_successful_order(agent)
    run_failed_payment_order(agent)


def run_travel_scenarios(config) -> None:
    """Run all travel booking scenarios."""
    logger.info("section=travel_booking")

    coordinator = TravelCoordinator(config)

    run_successful_trip(coordinator)
    run_failed_hotel_booking(coordinator)


def main() -> int:
    """Main entry point."""
    try:
        config = load_config()
        configure_logging(config)

        logger.info("react_agent_compensation_demo version=0.1.0")

        run_order_scenarios(config)
        run_travel_scenarios(config)

        logger.info("demo_completed")
        return 0

    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        logger.info("demo_interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
