# react-agent-compensation

A framework-agnostic compensation/rollback library for ReAct agents.

## Features

- **Framework-agnostic Core**: Works with any agent framework
- **LangChain Adaptor**: First-class LangChain/LangGraph integration
- **Compensation Patterns**: Automatic rollback on failures
- **Retry Strategies**: Exponential backoff with jitter
- **Dependency Tracking**: Topological sort for correct rollback order
- **MCP Integration**: Auto-discover compensation pairs from tool schemas

## Installation

```bash
pip install react-agent-compensation
```

With LangChain support:

```bash
pip install react-agent-compensation[langchain]
```

With LLM-based extraction:

```bash
pip install react-agent-compensation[llm]
```

## Quick Start

```python
from react_agent_compensation.langchain_adaptor import create_compensated_agent

agent = create_compensated_agent(
    model="gpt-4",
    tools=[book_flight, cancel_flight],
    compensation_mapping={"book_flight": "cancel_flight"},
)

result = agent.invoke({"messages": [("user", "Book me a flight to NYC")]})
```

## Core Components

### RecoveryManager

The brain of the compensation system:

```python
from react_agent_compensation.core import RecoveryManager

manager = RecoveryManager(
    compensation_pairs={"book_flight": "cancel_flight"},
    alternative_map={"book_flight": ["book_flight_backup"]},
)

# Record before execution
record = manager.record_action("book_flight", {"dest": "NYC"})

# Mark complete on success
manager.mark_completed(record.id, result={"booking_id": "123"})

# Rollback on failure
manager.rollback()
```

### Extraction Strategies

Multiple strategies for extracting compensation parameters:

- **State Mappers**: Custom lambda functions
- **Schema-based**: Declarative path mappings
- **Heuristic**: Auto-detect common ID fields
- **LLM-based**: Use LLM for complex extraction

## License

MIT
