# react-agent-compensation

A framework-agnostic compensation/rollback library for ReAct agents. Automatically tracks actions and rolls them back when subsequent operations fail, ensuring data consistency in AI agent workflows.

[![PyPI version](https://badge.fury.io/py/react-agent-compensation.svg)](https://pypi.org/project/react-agent-compensation/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Framework-agnostic Core**: Works with any agent framework (LangChain, CrewAI, custom)
- **LangChain/LangGraph Integration**: First-class support with middleware and agent factories
- **MCP Integration**: Auto-discover compensation pairs from MCP server tool annotations
- **Automatic Rollback**: When an action fails, all previously completed actions are automatically compensated
- **Retry Strategies**: Configurable exponential backoff, linear backoff, and fixed delay
- **Dependency Tracking**: Topological sort ensures correct rollback order
- **Multi-Agent Support**: Shared transaction logs across multiple agents
- **Extraction Strategies**: Multiple ways to extract compensation parameters (schema, heuristic, LLM-based)
- **Strategic Context Preservation** *(v0.2.0)*: Tracks cumulative failures to help LLM make informed decisions about what to try next
- **Goal-Aware Recovery** *(v0.2.0)*: Reminds LLM of optimization objectives during replanning for holistic decision-making
- **Permanent Failure Detection** *(v0.2.0)*: Classifies errors as permanent vs transient to guide retry strategies

## Installation

```bash
# Core only
pip install react-agent-compensation

# With LangChain support
pip install react-agent-compensation[langchain]

# With MCP support (for Model Context Protocol servers)
pip install react-agent-compensation[mcp]

# With LLM-based extraction
pip install react-agent-compensation[llm]

# Everything
pip install react-agent-compensation[all]
```

## Quick Start

### Basic Usage with LangChain

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from react_agent_compensation.langchain_adaptor import CompensationMiddleware

# Define your tools
@tool
def book_flight(destination: str, date: str) -> dict:
    """Book a flight to a destination."""
    return {"booking_id": "FL123", "destination": destination, "date": date}

@tool
def cancel_flight(booking_id: str) -> dict:
    """Cancel a flight booking."""
    return {"cancelled": True, "booking_id": booking_id}

# Create middleware with compensation pairs
middleware = CompensationMiddleware(
    compensation_pairs={"book_flight": "cancel_flight"},
)

# Wrap your tools
wrapped_tools = middleware.wrap_tools([book_flight, cancel_flight])

# Use with LangGraph
from langgraph.prebuilt import create_react_agent

model = ChatOpenAI(model="gpt-4")
agent = create_react_agent(model, wrapped_tools)

# Run the agent - compensation is automatic on failures
result = agent.invoke({"messages": [("user", "Book a flight to NYC for tomorrow")]})

# Access transaction log
for record_id, record in middleware.log.snapshot().items():
    print(f"{record.action}: {record.status.value}")

# Manual rollback if needed
middleware.rollback()
```

### MCP Integration

Connect to MCP servers with automatic compensation discovery:

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from react_agent_compensation.langchain_adaptor import create_compensated_mcp_agent

# Create agent connected to MCP server
agent, client = await create_compensated_mcp_agent(
    model=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
    mcp_servers={
        "myserver": {
            "url": "http://localhost:8000/sse",
            "transport": "sse",
        }
    },
    system_prompt="You are a helpful assistant.",
)

# Compensation pairs are auto-discovered from server annotations
pairs = await client.get_compensation_pairs()
print(f"Discovered pairs: {pairs}")
# Output: {'add_item': 'delete_item', 'create_order': 'cancel_order'}

# Run the agent
result = await agent.ainvoke({
    "messages": [("user", "Add an item called Widget")]
})

# Rollback if needed
await client.rollback()
```

### Core Components

```python
from react_agent_compensation.core import RecoveryManager, RetryPolicy

# Create recovery manager
manager = RecoveryManager(
    compensation_pairs={
        "book_flight": "cancel_flight",
        "reserve_hotel": "cancel_hotel",
    },
    retry_policy=RetryPolicy(
        max_retries=3,
        initial_delay=1.0,
        backoff_multiplier=2.0,
    ),
)

# Record actions
record = manager.record_action("book_flight", {"dest": "NYC", "date": "2024-01-15"})

# Mark complete with result
manager.mark_completed(record.id, result={"booking_id": "FL123"})

# On failure, rollback all completed actions
rollback_result = manager.rollback()
print(f"Rolled back: {rollback_result.compensated}")
```

### Goal-Aware Recovery (v0.2.0)

Enable holistic replanning with optimization goals:

```python
from react_agent_compensation.langchain_adaptor import create_compensated_agent

# Create agent with optimization goals
agent = create_compensated_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[book_flight, cancel_flight, reserve_hotel, cancel_hotel],
    compensation_pairs={
        "book_flight": "cancel_flight",
        "reserve_hotel": "cancel_hotel",
    },
    goals=["minimize_total_cost", "prefer_direct_flights", "maximize_loyalty_points"],
)

# When failures occur, the LLM receives:
# 1. Cumulative failure context (what was tried and failed)
# 2. List of rolled-back actions that need re-doing
# 3. Reminder of optimization goals for holistic replanning
```

### Strategic Context Preservation (v0.2.0)

Access cumulative failure context programmatically:

```python
from react_agent_compensation.core import RecoveryManager

manager = RecoveryManager(compensation_pairs={"book": "cancel"})

# After failures, get summary for LLM context
failure_summary = manager.get_failure_summary()
# Returns formatted summary like:
# [PREVIOUS FAILED ATTEMPTS]
# book:
#   - Attempt 1: (dest=NYC) [PERMANENT]
#     Error: Flight unavailable due to weather
# Consider using different parameters or approaches.

# Access failure context directly
for attempt in manager.failure_context.attempts:
    print(f"{attempt.action}: {attempt.error} (permanent={attempt.is_permanent})")
```

## MCP Server Setup

Create an MCP server with compensation annotations:

```python
from fastmcp import FastMCP

mcp = FastMCP("My Server")

@mcp.tool(
    annotations={
        "x-compensation-pair": "delete_item",
        "x-action-type": "create",
    }
)
def add_item(name: str) -> dict:
    """Add an item to the database."""
    # Your implementation
    return {"id": "123", "name": name}

@mcp.tool(
    annotations={
        "x-compensation-pair": "add_item",
        "x-action-type": "delete",
    }
)
def delete_item(item_id: str) -> dict:
    """Delete an item from the database."""
    # Your implementation
    return {"deleted": True}

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
```

## Extraction Strategies

Multiple ways to extract parameters for compensation actions:

```python
from react_agent_compensation.langchain_adaptor import CompensationMiddleware

# Schema-based extraction
middleware = CompensationMiddleware(
    compensation_pairs={"book_flight": "cancel_flight"},
    compensation_schemas={
        "book_flight": {
            "cancel_flight": {
                "booking_id": "result.booking_id",
            }
        }
    },
)

# State mapper (custom function)
middleware = CompensationMiddleware(
    compensation_pairs={"book_flight": "cancel_flight"},
    state_mappers={
        "book_flight": lambda input, result: {"booking_id": result["booking_id"]},
    },
)

# Heuristic (auto-detect common ID fields)
from react_agent_compensation.core.extraction import HeuristicExtraction

manager = RecoveryManager(
    compensation_pairs={"book_flight": "cancel_flight"},
    extraction_strategy=HeuristicExtraction(),
)
```

## Examples

See the [examples](./examples) directory for complete working examples:

- **[LangChain Agent](./examples/langchain_agent.py)**: Basic LangChain integration with LangSmith tracing
- **[MCP Integration](./examples/mcp/)**: Full MCP server/client example with MongoDB backend

## Documentation

- [MCP Integration Guide](./docs/mcp_integration.md) - Complete guide for MCP server integration

## Architecture

```
┌─────────────────────────────────────────┐
│            Your Agent                    │
│    (LangChain, CrewAI, Custom)          │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│      CompensationMiddleware              │
│   or CompensatedMCPTool                  │
│  - Intercepts tool calls                 │
│  - Records actions in TransactionLog     │
│  - Triggers rollback on failure          │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│         RecoveryManager                  │
│  - Manages compensation pairs            │
│  - Handles retry logic                   │
│  - Executes rollback in correct order    │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│         TransactionLog                   │
│  - Stores action records                 │
│  - Tracks status and dependencies        │
│  - Supports multi-agent scenarios        │
└─────────────────────────────────────────┘
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `RecoveryManager` | Main orchestrator for compensation logic |
| `TransactionLog` | Stores and manages action records |
| `ActionRecord` | Individual action with status and metadata |
| `RetryPolicy` | Configuration for retry behavior |
| `FailureContext` | Tracks cumulative failures for Strategic Context Preservation |
| `FailedAttempt` | Record of a single failed attempt |
| `is_likely_permanent` | Heuristic to classify errors as permanent vs transient |

### LangChain Adaptor

| Class/Function | Description |
|----------------|-------------|
| `CompensationMiddleware` | Wraps tools with compensation tracking |
| `create_compensated_agent` | Factory for creating compensated agents |
| `create_multi_agent_log` | Shared log for multi-agent scenarios |

### MCP Integration

| Class/Function | Description |
|----------------|-------------|
| `MCPCompensationClient` | High-level MCP client with compensation |
| `CompensatedMCPTool` | Wrapped MCP tool with tracking |
| `create_compensated_mcp_agent` | Factory for MCP-based agents |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.
