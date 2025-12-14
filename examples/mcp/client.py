"""LangChain Agent with MCP Tools and Automatic Compensation.

This example demonstrates:
1. Connecting to an MCP server (Family Coordination)
2. Auto-discovering compensation pairs from tool annotations
3. Running a LangChain agent with compensated tools
4. Automatic rollback when operations fail

Prerequisites:
    1. MongoDB running on localhost:27017
    2. MCP server running: python -m examples.mcp.server
    3. Environment variables:
       - GOOGLE_API_KEY or OPENAI_API_KEY
       - LANGSMITH_API_KEY (optional, for tracing)

Run:
    python -m examples.mcp.client
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mcp_client")

# Suppress verbose library logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)


async def run_successful_scenario(agent, client):
    """Demonstrate successful family coordination operations."""
    print("\n" + "=" * 70)
    print("Scenario 1: Successful Operations")
    print("=" * 70)

    message = (
        "Add a family member named Sarah as Mom who is driving from Boston, "
        "arriving at 14:00. Then add a pickup task for her at Boston Airport."
    )
    print(f"\nUser: {message}\n")

    result = await agent.ainvoke(
        {"messages": [("user", message)]},
        config={
            "run_name": "successful_family_coordination",
            "tags": ["mcp", "family", "success"],
        },
    )

    # Show what was recorded
    print("\nTransaction log:")
    for record_id, record in client.recovery_manager.log.snapshot().items():
        print(f"  {record.action}: {record.status.value}")

    return result


async def run_failure_scenario(agent, client):
    """Demonstrate failure with automatic compensation."""
    print("\n" + "=" * 70)
    print("Scenario 2: Failure with Compensation")
    print("=" * 70)

    # First add a family member
    print("\nStep 1: Add a family member...")
    result1 = await agent.ainvoke(
        {"messages": [("user", "Add a family member named John as Dad")]},
        config={"run_name": "add_member"},
    )

    print("\nStep 2: Try to delete a non-existent member (will fail)...")
    # This will fail but we want to show the agent handles it
    result2 = await agent.ainvoke(
        {"messages": [("user", "Delete family member named NonExistent")]},
        config={"run_name": "delete_nonexistent"},
    )

    # Show transaction state
    print("\nTransaction log after operations:")
    for record_id, record in client.recovery_manager.log.snapshot().items():
        print(f"  {record.action}: {record.status.value}")


async def run_interactive_demo(agent, client):
    """Run an interactive demo with the agent."""
    print("\n" + "=" * 70)
    print("Interactive Demo - Family Coordination")
    print("=" * 70)
    print("\nCommands: 'quit' to exit, 'status' for transaction log, 'rollback' to rollback\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.lower() == "quit":
                break

            if user_input.lower() == "status":
                print("\nTransaction log:")
                for record_id, record in client.recovery_manager.log.snapshot().items():
                    print(f"  [{record.id[:8]}] {record.action}: {record.status.value}")
                continue

            if user_input.lower() == "rollback":
                print("\nExecuting rollback...")
                result = await client.rollback()
                print(f"Rollback result: {result}")
                continue

            # Run the agent
            result = await agent.ainvoke(
                {"messages": [("user", user_input)]},
            )

            # Print the response
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    print(f"\nAgent: {last_msg.content}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")


async def main():
    """Main entry point."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    from react_agent_compensation.langchain_adaptor import create_compensated_mcp_agent

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: GOOGLE_API_KEY or OPENAI_API_KEY must be set")
        sys.exit(1)

    print("\nFamily Coordination MCP Client")
    print("Using react-agent-compensation with MCP integration\n")

    # Configure LangSmith tracing (optional)
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "mcp-compensation-demo")
        print("LangSmith tracing enabled")

    # Create model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
    )

    print("\nConnecting to MCP server at http://localhost:8000/sse...")

    try:
        # Create compensated MCP agent
        agent, client = await create_compensated_mcp_agent(
            model=model,
            mcp_servers={
                "family": {
                    "url": "http://localhost:8000/sse",
                    "transport": "sse",
                }
            },
            system_prompt="""You are a family coordination assistant helping plan
            family gatherings. You can:
            - Add and manage family members
            - Create tasks and pickups
            - Manage cooking schedules

            When operations fail, explain what happened clearly.""",
        )

        # Show discovered compensation pairs
        pairs = await client.get_compensation_pairs()
        print(f"\nDiscovered {len(pairs)} compensation pairs:")
        for forward, compensator in pairs.items():
            print(f"  {forward} -> {compensator}")

        # Run demo scenarios
        if "--interactive" in sys.argv:
            await run_interactive_demo(agent, client)
        else:
            await run_successful_scenario(agent, client)
            client.recovery_manager.clear()  # Clear log between scenarios
            await run_failure_scenario(agent, client)

        print("\n" + "=" * 70)
        print("Demo completed")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("  1. MongoDB is running on localhost:27017")
        print("  2. MCP server is running: python -m examples.mcp.server")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
