"""Order processing agent with automatic compensation."""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI

from react_agent_compensation.core import CompensationSchema
from react_agent_compensation.langchain_adaptor import create_compensated_agent

from examples.config import Config
from examples.tools.inventory import reserve_inventory, release_inventory
from examples.tools.orders import create_order, cancel_order
from examples.tools.payments import process_payment, refund_payment
from examples.tools.shipping import create_shipment, cancel_shipment

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    """Request to process an order."""

    customer_id: str
    items: str
    amount: float
    payment_method: str
    shipping_address: str


@dataclass
class OrderResult:
    """Result of order processing."""

    success: bool
    order_id: str = ""
    error: str = ""
    compensated_actions: List[str] = None

    def __post_init__(self):
        if self.compensated_actions is None:
            self.compensated_actions = []

    @property
    def compensated_count(self) -> int:
        return len(self.compensated_actions)


class OrderProcessorAgent:
    """E-commerce order processing agent with automatic compensation.

    Handles the complete order workflow:
    1. Reserve inventory
    2. Create order
    3. Process payment
    4. Create shipment

    If any step fails, all previous steps are automatically compensated.
    """

    SYSTEM_PROMPT = """You are an e-commerce order processing assistant.
When processing an order, execute these steps in sequence:
1. Reserve inventory for the products
2. Create the order record
3. Process the payment
4. Create the shipment

Use the available tools to complete each step. If any step fails,
the system will automatically compensate previous successful steps."""

    def __init__(self, config: Config):
        """Initialize the order processor agent.

        Args:
            config: Application configuration.
        """
        self._config = config
        self._agent = self._build_agent()

    def _build_agent(self):
        """Build the agent with compensation configuration."""
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=self._config.google_api_key,
        )

        # Define compensation mapping: action -> compensation action
        compensation_mapping = {
            "reserve_inventory": "release_inventory",
            "create_order": "cancel_order",
            "process_payment": "refund_payment",
            "create_shipment": "cancel_shipment",
        }

        # Define parameter extraction schemas
        compensation_schemas = {
            "reserve_inventory": CompensationSchema(
                param_mapping={"reservation_id": "result.reservation_id"}
            ),
            "create_order": CompensationSchema(
                param_mapping={"order_id": "result.order_id"}
            ),
            "process_payment": CompensationSchema(
                param_mapping={
                    "payment_id": "result.payment_id",
                    "transaction_id": "result.transaction_id",
                }
            ),
            "create_shipment": CompensationSchema(
                param_mapping={"shipment_id": "result.shipment_id"}
            ),
        }

        # Note: error_strategies not exposed in new API - using default detection
        return create_compensated_agent(
            model=model,
            tools=[
                reserve_inventory,
                release_inventory,
                create_order,
                cancel_order,
                process_payment,
                refund_payment,
                create_shipment,
                cancel_shipment,
            ],
            compensation_mapping=compensation_mapping,
            compensation_schemas=compensation_schemas,
            system_prompt=self.SYSTEM_PROMPT,
        )

    def process_order(self, request: OrderRequest) -> OrderResult:
        """Process an order with automatic rollback on failure.

        Args:
            request: Order processing request.

        Returns:
            OrderResult with success status or compensation details.
        """
        message = (
            f"Process an order for customer {request.customer_id} "
            f"with products {request.items}. "
            f"The order total is ${request.amount:.2f} "
            f"to be paid with {request.payment_method}. "
            f"Ship to {request.shipping_address}."
        )

        config = {
            "run_name": "order_processing",
            "tags": ["ecommerce", "order"],
            "metadata": {
                "customer_id": request.customer_id,
                "amount": request.amount,
            },
        }

        try:
            result = self._agent.invoke(
                {"messages": [("user", message)]},
                config=config,
            )

            # Check for compensation in the result
            messages = result.get("messages", [])
            compensated = self._extract_compensation_actions(messages)

            if compensated:
                return OrderResult(
                    success=False,
                    error="Order failed, actions compensated",
                    compensated_actions=compensated,
                )

            # Extract order ID from successful result
            order_id = self._extract_order_id(messages)
            return OrderResult(success=True, order_id=order_id)

        except Exception as e:
            logger.error("order_processing error=%s", str(e))
            return OrderResult(success=False, error=str(e))

    def _extract_compensation_actions(self, messages: List[Any]) -> List[str]:
        """Extract names of compensation actions that were executed."""
        compensated = []
        compensation_tools = {
            "release_inventory",
            "cancel_order",
            "refund_payment",
            "cancel_shipment",
        }

        for msg in messages:
            if hasattr(msg, "name") and msg.name in compensation_tools:
                compensated.append(msg.name)

        return compensated

    def _extract_order_id(self, messages: List[Any]) -> str:
        """Extract order ID from message history."""
        import json

        for msg in messages:
            if not hasattr(msg, "content"):
                continue

            content = msg.content

            # Parse JSON string if needed
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    continue

            if isinstance(content, dict) and "order_id" in content:
                return content["order_id"]

        return ""
