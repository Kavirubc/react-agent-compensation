"""Basic usage example for react-agent-compensation.

This example demonstrates the core compensation system without
any LLM/agent framework. It shows:
- Recording actions with compensators
- Dependency tracking
- Rollback with topological ordering
"""

from react_agent_compensation.core import (
    ActionStatus,
    RecoveryManager,
    RetryPolicy,
)


def simulate_book_flight(params: dict) -> dict:
    """Simulate booking a flight."""
    print(f"  Booking flight to {params['destination']}...")
    return {"booking_id": "FL-12345", "status": "confirmed"}


def simulate_cancel_flight(params: dict) -> dict:
    """Simulate canceling a flight."""
    print(f"  Canceling flight {params.get('booking_id', 'unknown')}...")
    return {"status": "cancelled"}


def simulate_book_hotel(params: dict) -> dict:
    """Simulate booking a hotel."""
    print(f"  Booking hotel in {params['city']}...")
    return {"reservation_id": "HT-67890", "status": "confirmed"}


def simulate_cancel_hotel(params: dict) -> dict:
    """Simulate canceling a hotel."""
    print(f"  Canceling hotel {params.get('reservation_id', 'unknown')}...")
    return {"status": "cancelled"}


def simulate_payment_failure(params: dict) -> dict:
    """Simulate a payment that fails."""
    print(f"  Processing payment of ${params['amount']}...")
    raise RuntimeError("Payment declined: insufficient funds")


# Custom executor to run our simulated tools
class SimpleExecutor:
    def __init__(self):
        self.tools = {
            "cancel_flight": simulate_cancel_flight,
            "cancel_hotel": simulate_cancel_hotel,
        }

    def execute(self, action: str, params: dict):
        if action in self.tools:
            return self.tools[action](params)
        raise ValueError(f"Unknown action: {action}")


def main():
    print("=" * 60)
    print("React Agent Compensation - Basic Example")
    print("=" * 60)

    # Create recovery manager with compensation pairs
    manager = RecoveryManager(
        compensation_pairs={
            "book_flight": "cancel_flight",
            "book_hotel": "cancel_hotel",
        },
        retry_policy=RetryPolicy(max_retries=2, initial_delay=0.1),
        action_executor=SimpleExecutor(),
    )

    try:
        # Step 1: Book a flight
        print("\n[Step 1] Booking flight...")
        rec1 = manager.record_action("book_flight", {"destination": "NYC"})
        result1 = simulate_book_flight({"destination": "NYC"})
        manager.mark_completed(rec1.id, result1)
        print(f"  Success! Booking ID: {result1['booking_id']}")

        # Step 2: Book a hotel (depends on flight)
        print("\n[Step 2] Booking hotel...")
        rec2 = manager.record_action("book_hotel", {
            "city": "NYC",
            "linked_booking": result1['booking_id']
        })
        result2 = simulate_book_hotel({"city": "NYC"})
        manager.mark_completed(rec2.id, result2)
        print(f"  Success! Reservation ID: {result2['reservation_id']}")

        # Step 3: Process payment (will fail)
        print("\n[Step 3] Processing payment...")
        rec3 = manager.record_action("process_payment", {"amount": 500})
        result3 = simulate_payment_failure({"amount": 500})
        manager.mark_completed(rec3.id, result3)

    except Exception as e:
        print(f"  FAILED: {e}")

        # Trigger rollback
        print("\n[Rollback] Rolling back completed actions...")
        print("-" * 40)

        rollback_result = manager.rollback()

        print("-" * 40)
        print(f"\nRollback Summary:")
        print(f"  Success: {rollback_result.success}")
        print(f"  Compensated: {len(rollback_result.compensated)} actions")
        print(f"  Message: {rollback_result.message}")

    # Show transaction log state
    print("\n[Transaction Log]")
    for record_id, record in manager.log.snapshot().items():
        print(f"  {record.action}: {record.status}")

    print("\n" + "=" * 60)
    print("Example completed!")


if __name__ == "__main__":
    main()
