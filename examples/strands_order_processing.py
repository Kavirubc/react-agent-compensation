"""
Order Processing with Automatic Compensation using AWS Strands

This example demonstrates an order processing workflow using AWS Strands
Agents SDK with the react-agent-compensation framework. Features:

1. Multi-step order workflow (inventory, payment)
2. Automatic compensation on failure
3. CompensationSchema for declarative parameter mapping
4. Async support with invoke_async()
5. Goal-Aware Recovery guidance

The workflow:
- Reserve inventory
- Process payment
- If payment fails, inventory reservation is automatically released

Requirements:
    pip install strands-agents react-agent-compensation
"""

import asyncio
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


# Simulated ID counters
_id_counter = {"reservation": 0, "payment": 0}


# Mock tool implementations
def reserve_inventory_impl(product_ids: list, quantity: int = 1) -> str:
    """Reserve inventory for products."""
    _id_counter["reservation"] += 1
    reservation_id = f"RES-{_id_counter['reservation']:04d}"
    products_str = ", ".join(product_ids) if isinstance(product_ids, list) else product_ids
    logger.info(f"Reserved inventory for {products_str}, qty={quantity}, ID: {reservation_id}")
    return f'{{"reservation_id": "{reservation_id}", "products": {product_ids}, "quantity": {quantity}, "status": "reserved"}}'


def release_inventory_impl(reservation_id: str) -> str:
    """Release reserved inventory."""
    logger.info(f"Released inventory reservation: {reservation_id}")
    return f'{{"released": true, "reservation_id": "{reservation_id}"}}'


def process_payment_impl(amount: float, method: str = "credit_card") -> str:
    """Process payment for order."""
    _id_counter["payment"] += 1
    payment_id = f"PAY-{_id_counter['payment']:04d}"

    # Simulate failure for high amounts
    if amount > 10000:
        logger.warning(f"Payment failed: Amount ${amount} exceeds limit")
        return f'{{"error": "Payment amount ${amount} exceeds the $10,000 limit"}}'

    logger.info(f"Processed payment of ${amount} via {method}, ID: {payment_id}")
    return f'{{"payment_id": "{payment_id}", "amount": {amount}, "method": "{method}", "status": "completed"}}'


def refund_payment_impl(payment_id: str) -> str:
    """Refund a payment."""
    logger.info(f"Refunded payment: {payment_id}")
    return f'{{"refunded": true, "payment_id": "{payment_id}"}}'


def run_example():
    """Run the Strands order processing example."""
    # Import Strands - will fail if not installed
    try:
        from strands import tool
    except ImportError:
        print("\nStrands is not installed. Install with: pip install strands-agents")
        print("Showing example code structure instead.\n")
        show_example_structure()
        return

    # Import compensation framework
    from react_agent_compensation.strands_adaptor import (
        create_compensated_agent,
        RetryPolicy,
    )
    from react_agent_compensation.core.extraction import CompensationSchema

    # Define tools using Strands @tool decorator
    @tool
    def reserve_inventory(product_ids: list, quantity: int = 1) -> str:
        """Reserve inventory for products before processing order."""
        return reserve_inventory_impl(product_ids, quantity)

    @tool
    def release_inventory(reservation_id: str) -> str:
        """Release previously reserved inventory."""
        return release_inventory_impl(reservation_id)

    @tool
    def process_payment(amount: float, method: str = "credit_card") -> str:
        """Process payment for the order."""
        return process_payment_impl(amount, method)

    @tool
    def refund_payment(payment_id: str) -> str:
        """Refund a previously processed payment."""
        return refund_payment_impl(payment_id)

    # Define compensation schemas
    compensation_schemas = {
        "reserve_inventory": CompensationSchema(
            param_mapping={"reservation_id": "result.reservation_id"}
        ),
        "process_payment": CompensationSchema(
            param_mapping={"payment_id": "result.payment_id"}
        ),
    }

    # Create compensated agent
    agent = create_compensated_agent(
        system_prompt=(
            "You are an order processing assistant. When processing an order:\n"
            "1. First reserve inventory for the requested products\n"
            "2. Then process the payment\n"
            "Always complete both steps in sequence."
        ),
        tools=[reserve_inventory, release_inventory, process_payment, refund_payment],
        compensation_mapping={
            "reserve_inventory": "release_inventory",
            "process_payment": "refund_payment",
        },
        compensation_schemas=compensation_schemas,
        retry_policy=RetryPolicy(max_retries=2, initial_delay=0.5),
        goals=["fast_processing", "minimize_failures"],
    )

    # Scenario 1: Successful order
    print("=" * 70)
    print("Scenario 1: Successful Order Processing")
    print("=" * 70)

    result = agent("Process an order for products SKU001, SKU002 with payment of $150")
    print(f"\nResult: {result}")

    # Scenario 2: Order that triggers compensation
    print("\n" + "=" * 70)
    print("Scenario 2: Failed Payment with Automatic Compensation")
    print("=" * 70)
    print("Note: Payment will fail due to amount > $10,000, triggering rollback")

    result = agent("Process an order for products SKU003, SKU004 with payment of $15000")
    print(f"\nResult: {result}")

    print("\n" + "=" * 70)
    print("Example completed")
    print("=" * 70)


async def run_example_async():
    """Run the async version of the example."""
    # Import Strands - will fail if not installed
    try:
        from strands import tool
    except ImportError:
        print("\nStrands is not installed for async example.")
        return

    # Import compensation framework
    from react_agent_compensation.strands_adaptor import (
        create_compensated_agent,
        RetryPolicy,
    )
    from react_agent_compensation.core.extraction import CompensationSchema

    # Define tools
    @tool
    def reserve_inventory(product_ids: list, quantity: int = 1) -> str:
        """Reserve inventory for products before processing order."""
        return reserve_inventory_impl(product_ids, quantity)

    @tool
    def release_inventory(reservation_id: str) -> str:
        """Release previously reserved inventory."""
        return release_inventory_impl(reservation_id)

    @tool
    def process_payment(amount: float, method: str = "credit_card") -> str:
        """Process payment for the order."""
        return process_payment_impl(amount, method)

    @tool
    def refund_payment(payment_id: str) -> str:
        """Refund a previously processed payment."""
        return refund_payment_impl(payment_id)

    # Define compensation schemas
    compensation_schemas = {
        "reserve_inventory": CompensationSchema(
            param_mapping={"reservation_id": "result.reservation_id"}
        ),
        "process_payment": CompensationSchema(
            param_mapping={"payment_id": "result.payment_id"}
        ),
    }

    # Create compensated agent
    agent = create_compensated_agent(
        system_prompt="You are an order processing assistant.",
        tools=[reserve_inventory, release_inventory, process_payment, refund_payment],
        compensation_mapping={
            "reserve_inventory": "release_inventory",
            "process_payment": "refund_payment",
        },
        compensation_schemas=compensation_schemas,
        retry_policy=RetryPolicy(max_retries=1),
        goals=["fast_processing"],
    )

    print("=" * 70)
    print("Async Example: Order Processing")
    print("=" * 70)

    # Use invoke_async for async execution
    result = await agent.invoke_async("Process order for SKU005 with payment of $200")
    print(f"\nAsync Result: {result}")


def show_example_structure():
    """Show the example structure when Strands is not installed."""
    print("""
Example Code Structure:

1. Define compensation mapping:
   compensation_mapping = {
       "reserve_inventory": "release_inventory",
       "process_payment": "refund_payment",
   }

2. Define parameter extraction schemas:
   compensation_schemas = {
       "reserve_inventory": CompensationSchema(
           param_mapping={"reservation_id": "result.reservation_id"}
       ),
       "process_payment": CompensationSchema(
           param_mapping={"payment_id": "result.payment_id"}
       ),
   }

3. Create compensated agent:
   agent = create_compensated_agent(
       system_prompt="You are an order processing assistant.",
       tools=[reserve_inventory, release_inventory, process_payment, refund_payment],
       compensation_mapping=compensation_mapping,
       compensation_schemas=compensation_schemas,
       retry_policy=RetryPolicy(max_retries=2),
       goals=["fast_processing", "minimize_failures"],
   )

4. Execute (sync):
   result = agent("Process the order")

   Or async:
   result = await agent.invoke_async("Process the order")

When a tool fails:
- Automatic retry attempts with exponential backoff
- Alternative tool fallback if configured
- Rollback of all completed compensatable actions
- Result modified with compensation message
- State persisted to invocation_state
- Strategic Context Preservation and Goal-Aware Recovery
""")


if __name__ == "__main__":
    run_example()
