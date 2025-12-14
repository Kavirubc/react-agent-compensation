# MCP Integration Guide

This guide covers integrating `react-agent-compensation` with MCP (Model Context Protocol) servers to enable automatic compensation and rollback for LangChain agents using MCP tools.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Server Setup](#server-setup)
- [Client Setup](#client-setup)
- [Compensation Annotations](#compensation-annotations)
- [API Reference](#api-reference)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Overview

### What is MCP?

MCP (Model Context Protocol) is an open protocol that standardizes how AI applications connect to external tools and data sources. It provides:

- **Standardized tool interfaces** - Tools are defined with JSON schemas
- **Multiple transports** - SSE, stdio, and more
- **Language-agnostic** - Works with any language that implements the protocol

### Why MCP + Compensation?

When AI agents use MCP tools to perform actions (creating records, modifying data, etc.), failures can leave the system in an inconsistent state. The compensation pattern automatically rolls back completed actions when later operations fail.

**Example scenario:**
1. Agent adds a family member to the database
2. Agent tries to add a pickup task but fails (invalid data)
3. **Without compensation**: Family member remains in database (inconsistent state)
4. **With compensation**: Family member is automatically deleted (rollback)

## Architecture

```
┌─────────────────────────────────────────┐
│        LangChain Agent (LangGraph)       │
│        create_react_agent()              │
└────────────────────┬────────────────────┘
                     │ Tool calls
┌────────────────────▼────────────────────┐
│         CompensatedMCPTool               │
│  - Records actions in TransactionLog     │
│  - Detects errors in responses           │
│  - Triggers rollback on failure          │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│         MCPCompensationClient            │
│  - Wraps langchain-mcp-adapters          │
│  - Discovers compensation pairs          │
│  - Manages RecoveryManager               │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│   MultiServerMCPClient (langchain)       │
│   - MCP protocol handling                │
│   - SSE/stdio transport                  │
└────────────────────┬────────────────────┘
                     │ HTTP/SSE
┌────────────────────▼────────────────────┐
│          FastMCP Server                  │
│  - Tools with x-compensation-pair        │
│  - Your application logic                │
└─────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
pip install react-agent-compensation[mcp,langchain]
```

### 1. Create MCP Server with Compensation Annotations

```python
# server.py
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
    # Your logic here
    return {"id": "123", "name": name}

@mcp.tool(
    annotations={
        "x-compensation-pair": "add_item",
        "x-action-type": "delete",
    }
)
def delete_item(item_id: str) -> dict:
    """Delete an item from the database."""
    # Your logic here
    return {"deleted": True}

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
```

### 2. Create LangChain Agent with Compensation

```python
# client.py
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from react_agent_compensation.langchain_adaptor import create_compensated_mcp_agent

async def main():
    # Create agent with compensation support
    agent, client = await create_compensated_mcp_agent(
        model=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
        mcp_servers={
            "myserver": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        },
    )

    # Show discovered compensation pairs
    pairs = await client.get_compensation_pairs()
    print(f"Compensation pairs: {pairs}")
    # Output: {'add_item': 'delete_item'}

    # Run agent
    result = await agent.ainvoke({
        "messages": [("user", "Add an item named 'Widget'")]
    })

    # View transaction log
    for record_id, record in client.recovery_manager.log.snapshot().items():
        print(f"{record.action}: {record.status.value}")

    # Manual rollback if needed
    # await client.rollback()

asyncio.run(main())
```

### 3. Run

```bash
# Terminal 1: Start server
python server.py

# Terminal 2: Run client
python client.py
```

## Server Setup

### FastMCP Installation

```bash
pip install fastmcp
```

### Basic Server Structure

```python
from fastmcp import FastMCP

mcp = FastMCP("Your Server Name")

# Define tools with @mcp.tool decorator
@mcp.tool()
def your_tool(param: str) -> dict:
    """Tool description."""
    return {"result": "value"}

# Run with SSE transport (recommended for HTTP)
if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
```

### Supported Transports

| Transport | Use Case | URL Format |
|-----------|----------|------------|
| SSE | HTTP servers, remote access | `http://host:port/sse` |
| stdio | Local processes, subprocess | N/A (use command) |

## Client Setup

### Installation

```bash
pip install react-agent-compensation[mcp,langchain]
```

This installs:
- `langchain-mcp-adapters` - Official LangChain MCP integration
- `mcp` - MCP protocol library
- `fastmcp` - FastMCP server framework

### Basic Client

```python
from react_agent_compensation.langchain_adaptor import create_compensated_mcp_agent

agent, client = await create_compensated_mcp_agent(
    model=your_model,
    mcp_servers={
        "server_name": {
            "url": "http://localhost:8000/sse",
            "transport": "sse",
        }
    },
)
```

### Multiple Servers

```python
agent, client = await create_compensated_mcp_agent(
    model=your_model,
    mcp_servers={
        "family": {
            "url": "http://localhost:8000/sse",
            "transport": "sse",
        },
        "inventory": {
            "url": "http://localhost:8001/sse",
            "transport": "sse",
        },
    },
)
```

### With System Prompt

```python
agent, client = await create_compensated_mcp_agent(
    model=your_model,
    mcp_servers={...},
    system_prompt="""You are a helpful assistant that manages
    family coordination. When operations fail, explain clearly.""",
)
```

## Compensation Annotations

### x-compensation-pair

Declares the tool that can undo this tool's action.

```python
@mcp.tool(
    annotations={
        "x-compensation-pair": "delete_item",  # Tool to call for rollback
    }
)
def add_item(name: str) -> dict:
    ...
```

Both tools should reference each other:

```python
# Forward action
@mcp.tool(annotations={"x-compensation-pair": "delete_item"})
def add_item(name: str) -> dict:
    return {"id": "123", "name": name}

# Compensation action
@mcp.tool(annotations={"x-compensation-pair": "add_item"})
def delete_item(item_id: str) -> dict:
    return {"deleted": True}
```

### x-action-type

Indicates the type of operation for logging and filtering.

| Value | Description |
|-------|-------------|
| `create` | Creates a new resource |
| `delete` | Removes a resource |
| `update` | Modifies existing resource |
| `read` | Query/fetch operation (not logged) |

```python
@mcp.tool(
    annotations={
        "x-compensation-pair": "delete_item",
        "x-action-type": "create",
    }
)
def add_item(name: str) -> dict:
    ...
```

### x-reversible

Marks update operations that preserve previous state for reversal.

```python
@mcp.tool(
    annotations={
        "x-action-type": "update",
        "x-reversible": True,
    }
)
def update_status(item_id: str, status: str) -> dict:
    previous = get_item(item_id)
    # Update logic...
    return {
        "message": "Updated",
        "previous_status": previous["status"],  # Store for potential reversal
    }
```

### x-category

Groups related tools for organization.

```python
@mcp.tool(
    annotations={
        "x-category": "inventory",
        "x-action-type": "create",
    }
)
def add_item(name: str) -> dict:
    ...
```

### x-destructive

Marks dangerous operations that cannot be undone.

```python
@mcp.tool(
    annotations={
        "x-action-type": "delete",
        "x-destructive": True,
        "x-requires-confirmation": True,
    }
)
def reset_database() -> dict:
    """Clear all data. Cannot be undone."""
    ...
```

## API Reference

### create_compensated_mcp_agent

Factory function to create a LangChain agent with MCP tools and compensation.

```python
async def create_compensated_mcp_agent(
    model: Any,
    mcp_servers: dict[str, dict[str, Any]],
    *,
    system_prompt: str | None = None,
    retry_policy: RetryPolicy | None = None,
    auto_rollback: bool = True,
    rollback_on_error: bool = True,
) -> tuple[CompiledGraph, MCPCompensationClient]:
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `Any` | LangChain chat model (ChatOpenAI, ChatGoogleGenerativeAI, etc.) |
| `mcp_servers` | `dict` | Server configuration mapping |
| `system_prompt` | `str` | Optional system prompt for the agent |
| `retry_policy` | `RetryPolicy` | Retry configuration for failed operations |
| `auto_rollback` | `bool` | Whether to enable automatic rollback (default: True) |
| `rollback_on_error` | `bool` | Trigger rollback on tool errors (default: True) |

**Returns:**
- `tuple[CompiledGraph, MCPCompensationClient]` - The agent and client instance

### MCPCompensationClient

High-level client for MCP with compensation support.

```python
class MCPCompensationClient:
    async def connect(self) -> None
    async def get_tools(self) -> list[CompensatedMCPTool]
    async def get_compensation_pairs(self) -> dict[str, str]
    async def rollback(self) -> RollbackResult

    @property
    def recovery_manager(self) -> RecoveryManager
```

**Methods:**

| Method | Description |
|--------|-------------|
| `connect()` | Connect to MCP servers and discover compensation pairs |
| `get_tools()` | Get wrapped tools with compensation tracking |
| `get_compensation_pairs()` | Get discovered forward→compensation mappings |
| `rollback()` | Execute rollback of all completed actions |

### CompensatedMCPTool

Wrapper around MCP tools that adds compensation tracking.

```python
class CompensatedMCPTool:
    @property
    def name(self) -> str
    @property
    def description(self) -> str
    @property
    def is_compensatable(self) -> bool
    @property
    def compensation_tool(self) -> str | None

    def invoke(self, input: dict, config: dict | None = None) -> str
    async def ainvoke(self, input: dict, config: dict | None = None) -> str
```

### MCPToolError

Exception raised when an MCP tool returns an error.

```python
class MCPToolError(Exception):
    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        result: Any = None,
    ):
        ...
```

## Advanced Usage

### Manual Compensation Control

```python
agent, client = await create_compensated_mcp_agent(
    model=model,
    mcp_servers={...},
    auto_rollback=False,  # Disable automatic rollback
)

# Run agent
result = await agent.ainvoke({"messages": [("user", "...")]})

# Check for failures
log = client.recovery_manager.log.snapshot()
failed = [r for r in log.values() if r.status == ActionStatus.FAILED]

if failed:
    # Decide whether to rollback
    user_choice = input("Rollback? (y/n): ")
    if user_choice == "y":
        await client.rollback()
```

### Custom Retry Policy

```python
from react_agent_compensation.core.retry import RetryPolicy

policy = RetryPolicy(
    max_retries=3,
    delay_seconds=1.0,
    backoff_multiplier=2.0,
    max_delay_seconds=30.0,
)

agent, client = await create_compensated_mcp_agent(
    model=model,
    mcp_servers={...},
    retry_policy=policy,
)
```

### Transaction Log Access

```python
# Get snapshot of all records
log = client.recovery_manager.log.snapshot()

for record_id, record in log.items():
    print(f"ID: {record.id}")
    print(f"Action: {record.action}")
    print(f"Status: {record.status.value}")
    print(f"Input: {record.input}")
    print(f"Output: {record.output}")
    print(f"Timestamp: {record.timestamp}")
    print("---")
```

### Clearing Transaction History

```python
# Clear all records (after successful completion)
client.recovery_manager.clear()
```

### Using Low-Level Client

For more control, use `MCPCompensationClient` directly:

```python
from react_agent_compensation.core.mcp import MCPCompensationClient

client = MCPCompensationClient(
    server_config={
        "myserver": {
            "url": "http://localhost:8000/sse",
            "transport": "sse",
        }
    },
)

await client.connect()

# Get tools and use them manually
tools = await client.get_tools()
tool = next(t for t in tools if t.name == "add_item")
result = tool.invoke({"name": "Widget"})

# Rollback if needed
await client.rollback()
```

## Troubleshooting

### Connection Refused

**Error:** `Connection refused at http://localhost:8000/sse`

**Solution:**
1. Ensure MCP server is running: `python server.py`
2. Check port is correct and not blocked by firewall
3. Verify URL format includes `/sse` for SSE transport

### No Compensation Pairs Found

**Error:** `Discovered 0 compensation pairs`

**Solution:**
1. Verify tools have `x-compensation-pair` in annotations:
   ```python
   @mcp.tool(annotations={"x-compensation-pair": "delete_item"})
   ```
2. Check server is returning tool schemas with annotations
3. Debug with:
   ```python
   tools = await client.get_tools()
   for t in tools:
       print(f"{t.name}: compensatable={t.is_compensatable}")
   ```

### Module Not Found

**Error:** `No module named 'langchain_mcp_adapters'`

**Solution:**
```bash
pip install langchain-mcp-adapters
# or
pip install react-agent-compensation[mcp]
```

### MongoDB Connection Failed

**Error:** `ServerSelectionTimeoutError`

**Solution (if using MongoDB backend):**
```bash
# macOS with Homebrew
brew services start mongodb-community

# Docker
docker run -d -p 27017:27017 mongo

# Verify
mongosh --eval "db.version()"
```

### Tool Returns Error But No Rollback

**Problem:** Tool returns `{"error": "..."}` but rollback doesn't trigger.

**Solution:** Ensure error format is detected. The client checks for:
- `"error"` key in response
- `"status": "error"` or `"status": "failed"`

If your error format differs, wrap your tools or adjust server responses.

### Async/Await Issues

**Error:** `RuntimeError: Event loop is already running`

**Solution:** Use `asyncio.run()` at the top level:
```python
import asyncio

async def main():
    agent, client = await create_compensated_mcp_agent(...)
    # ...

if __name__ == "__main__":
    asyncio.run(main())
```

In Jupyter notebooks, use:
```python
await main()  # Jupyter has its own event loop
```

## Complete Example

See the [MCP Family Coordination example](../examples/mcp/) for a complete working implementation with:

- 19 tools across 5 categories
- 4 compensation pairs
- MongoDB backend
- Interactive and demo modes

```bash
# Install dependencies
pip install react-agent-compensation[mcp,examples]

# Start MongoDB
docker run -d -p 27017:27017 mongo

# Start MCP server
python -m examples.mcp.server

# Run client (new terminal)
python -m examples.mcp.client
```

## Related Documentation

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- [MCP Specification](https://modelcontextprotocol.io/)
- [react-agent-compensation README](../README.md)
