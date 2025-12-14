# MCP Family Coordination Example

Demonstrates MCP (Model Context Protocol) integration with automatic compensation/rollback using `react-agent-compensation`.

## Overview

This example includes:
- **MCP Server** (`server.py`): FastMCP server with 19 tools for family coordination
- **MCP Client** (`client.py`): LangChain agent that connects to the server with automatic compensation

## Compensation Pairs

The server defines these compensation relationships via `x-compensation-pair` annotations:

| Forward Action | Compensation Action |
|----------------|---------------------|
| `add_family_member` | `delete_family_member` |
| `add_task` | `delete_task` |
| `add_pickup` | `delete_pickup` |
| `add_cooking_task` | `delete_cooking_task` |

When a forward action completes successfully but a later operation fails, all completed actions are automatically rolled back by calling their compensation actions.

## Prerequisites

1. **MongoDB**: Running on `localhost:27017`
   ```bash
   # macOS with Homebrew
   brew services start mongodb-community

   # Docker
   docker run -d -p 27017:27017 mongo
   ```

2. **Dependencies**:
   ```bash
   pip install react-agent-compensation[mcp,examples]
   ```

3. **Environment Variables**:
   ```bash
   # Required - choose one
   export GOOGLE_API_KEY="your-key"
   # or
   export OPENAI_API_KEY="your-key"

   # Optional - for LangSmith tracing
   export LANGSMITH_API_KEY="your-key"
   ```

## Running the Example

### Step 1: Start MongoDB
Ensure MongoDB is running on `localhost:27017`.

### Step 2: Start MCP Server
```bash
python -m examples.mcp.server
```

Output:
```
Starting Family Coordination MCP Server on http://0.0.0.0:8000/sse

Compensation pairs:
  add_family_member <-> delete_family_member
  add_task <-> delete_task
  add_pickup <-> delete_pickup
  add_cooking_task <-> delete_cooking_task

Requires MongoDB on localhost:27017
```

### Step 3: Run the Client
In a new terminal:
```bash
python -m examples.mcp.client
```

Or run in interactive mode:
```bash
python -m examples.mcp.client --interactive
```

## Example Output

```
Family Coordination MCP Client
Using react-agent-compensation with MCP integration

Connecting to MCP server at http://localhost:8000/sse...

Discovered 4 compensation pairs:
  add_family_member -> delete_family_member
  add_task -> delete_task
  add_pickup -> delete_pickup
  add_cooking_task -> delete_cooking_task

======================================================================
Scenario 1: Successful Operations
======================================================================

User: Add a family member named Sarah as Mom who is driving from Boston...

Transaction log:
  add_family_member: completed
  add_pickup: completed

======================================================================
Demo completed
======================================================================
```

## How It Works

### Server Side (FastMCP)

Tools declare compensation relationships via annotations:

```python
@mcp.tool(
    annotations={
        "x-compensation-pair": "delete_family_member",
        "x-category": "family",
        "x-action-type": "create",
    }
)
def add_family_member(name: str, role: str, ...) -> dict:
    ...
```

### Client Side (LangChain + Compensation)

The client auto-discovers pairs and wraps tools:

```python
from react_agent_compensation.langchain_adaptor import create_compensated_mcp_agent

agent, client = await create_compensated_mcp_agent(
    model=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
    mcp_servers={
        "family": {"url": "http://localhost:8000/sse", "transport": "sse"}
    },
)

# Tools are automatically wrapped with compensation tracking
result = await agent.ainvoke({"messages": [("user", "Add member John")]})

# Access transaction log
log = client.recovery_manager.log.snapshot()

# Manual rollback if needed
await client.rollback()
```

## File Structure

```
examples/mcp/
├── __init__.py      # Package init
├── README.md        # This file
├── server.py        # FastMCP server with compensation annotations
├── client.py        # LangChain agent client
└── database.py      # MongoDB connection
```

## Tool Categories

| Category | Tools |
|----------|-------|
| Family | `add_family_member`, `delete_family_member`, `get_family_members`, `update_family_member_status` |
| Tasks | `add_task`, `delete_task`, `get_tasks`, `update_task_status` |
| Pickups | `add_pickup`, `delete_pickup`, `get_pickups`, `update_pickup` |
| Cooking | `add_cooking_task`, `delete_cooking_task`, `get_cooking_tasks`, `update_cooking_task` |
| Schedule | `get_full_schedule`, `reset_schedule`, `get_travel_times` |

## Troubleshooting

### "Connection refused" error
- Ensure MongoDB is running: `mongod` or check Docker container
- Ensure MCP server is running: `python -m examples.mcp.server`

### "No module named 'langchain_mcp_adapters'"
```bash
pip install langchain-mcp-adapters
```

### "GOOGLE_API_KEY must be set"
```bash
export GOOGLE_API_KEY="your-google-ai-api-key"
```

## Related Documentation

- [MCP Integration Guide](../../docs/mcp_integration.md)
- [react-agent-compensation README](../../README.md)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
