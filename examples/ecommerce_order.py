"""
E-commerce Order Processing with Automatic Compensation

This example demonstrates a complete e-commerce order processing workflow
using react-agent-compensation with the following features:

1. Automatic compensation on failure
2. CompensationSchema for declarative parameter mapping
3. Multi-step workflow with dependency-aware rollback
4. Error detection and handling

The workflow:
- Reserve inventory
- Create order
- Process payment
- Create shipment

If any step fails, all previous steps are automatically compensated.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

from react_agent_compensation.core import CompensationSchema
from react_agent_compensation.langchain_adaptor import create_compensated_agent

from examples.tools import (
    reserve_inventory, release_inventory,
    create_order, cancel_order,
    process_payment, refund_payment,
    create_shipment, cancel_shipment
)

# Set up logging to see compensation actions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable logging for compensation middleware
compensation_logger = logging.getLogger("react_agent_compensation")
compensation_logger.setLevel(logging.INFO)


load_dotenv()

# Configure LangSmith tracing
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "react-agent-compensation-examples"))

# Verify API key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable must be set")


def create_order_agent():
    """Create an agent configured for order processing with compensation."""

    # Define compensation mapping: action -> compensation action
    compensation_mapping = {
        "reserve_inventory": "release_inventory",
        "create_order": "cancel_order",
        "process_payment": "refund_payment",
        "create_shipment": "cancel_shipment",
    }

    # Define compensation schemas for declarative parameter extraction
    compensation_schemas = {
        "reserve_inventory": CompensationSchema(
            param_mapping={
                "reservation_id": "result.reservation_id"
            }
        ),
        "create_order": CompensationSchema(
            param_mapping={
                "order_id": "result.order_id"
            }
        ),
        "process_payment": CompensationSchema(
            param_mapping={
                "payment_id": "result.payment_id",
                "transaction_id": "result.transaction_id"
            }
        ),
        "create_shipment": CompensationSchema(
            param_mapping={
                "shipment_id": "result.shipment_id"
            }
        ),
    }

    # Initialize the language model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0
    )

    # Create agent with compensation middleware
    agent = create_compensated_agent(
        model=model,
        tools=[
            reserve_inventory, release_inventory,
            create_order, cancel_order,
            process_payment, refund_payment,
            create_shipment, cancel_shipment
        ],
        compensation_mapping=compensation_mapping,
        compensation_schemas=compensation_schemas,
        system_prompt=(
            "You are an e-commerce order processing assistant. "
            "When processing an order, you must: "
            "1. Reserve inventory for the products "
            "2. Create the order "
            "3. Process payment for the order "
            "4. Create shipment for the order "
            "Use the available tools to complete each step in sequence."
        )
    )

    return agent


def run_successful_order(agent):
    """Run a successful order processing workflow."""
    print("=" * 70)
    print("Scenario 1: Successful Order Processing")
    print("=" * 70)

    user_message = (
        "Process an order for customer CUST001 with products PROD1,PROD2,PROD3. "
        "The order total is $150.00 and should be paid with credit_card. "
        "Ship to 123 Main St, City, State 12345"
    )

    print(f"\nUser Request: {user_message}\n")

    result = agent.invoke(
        {"messages": [("user", user_message)]},
        config={
            "run_name": "successful_order_processing",
            "tags": ["ecommerce", "order", "success"],
            "metadata": {
                "scenario": "successful_order",
                "customer_id": "CUST001"
            }
        }
    )

    print("\nOrder processing completed successfully.")
    return result


def run_failed_payment_order(agent):
    """Run an order that fails at payment, triggering compensation."""
    print("\n" + "=" * 70)
    print("Scenario 2: Payment Failure with Automatic Compensation")
    print("=" * 70)

    user_message = (
        "Process an order for customer CUST002 with products PROD4,PROD5. "
        "The order total is $15000.00 and should be paid with credit_card. "
        "Ship to 456 Oak Ave, City, State 67890"
    )

    print(f"\nUser Request: {user_message}\n")
    print("Note: Payment will fail due to amount exceeding limit ($10,000)")
    print("This will trigger automatic compensation of all previous steps.\n")

    # Use traceable to ensure compensation actions are visible
    @traceable(name="payment_failure_scenario", run_type="chain")
    def run_with_compensation():
        return agent.invoke(
            {"messages": [("user", user_message)]},
            config={
                "run_name": "payment_failure_with_compensation",
                "tags": ["ecommerce", "order", "failure", "compensation"],
                "metadata": {
                    "scenario": "payment_failure",
                    "customer_id": "CUST002",
                    "failure_reason": "amount_exceeds_limit"
                }
            }
        )

    result = run_with_compensation()

    print("\nCompensation completed: All previous steps have been rolled back.")
    return result


def main():
    """Main execution function."""
    print("\nE-commerce Order Processing with Automatic Compensation")
    print("Using react-agent-compensation v0.1.0\n")

    # Create the agent
    agent = create_order_agent()

    # Run successful scenario
    run_successful_order(agent)

    # Run failure scenario with compensation
    run_failed_payment_order(agent)

    print("\n" + "=" * 70)
    print("Example completed successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()
