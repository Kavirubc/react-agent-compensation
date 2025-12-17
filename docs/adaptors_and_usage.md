# Framework Adaptors and Usage Guide

This guide explains how to adapt the `react-agent-compensation` core to different agent frameworks. The core is framework-agnostic and uses Protocol-based interfaces, making it straightforward to integrate with any Python agent framework.

## Table of Contents

- [Core Architecture](#core-architecture)
- [Adaptor Methods](#adaptor-methods)
- [Method A: Middleware/Filters/Callbacks](#method-a-middlewarefilterscallbacks)
- [Method B: Tool Wrapper/Decorator](#method-b-tool-wrapperdecorator)
- [Method C: External Compensation Service](#method-c-external-compensation-service)
- [Framework-Specific Examples](#framework-specific-examples)
- [Best Practices](#best-practices)

## Core Architecture

The `react-agent-compensation` core is built around three key components that are **framework-agnostic**:

### 1. RecoveryManager

The "brain" of the compensation system. Handles:

- Recording actions before execution
- Retry logic with configurable backoff
- Alternative action fallback
- Rollback orchestration with dependency ordering

```python
from react_agent_compensation.core import RecoveryManager, CompensationPairs, RetryPolicy

manager = RecoveryManager(
    compensation_pairs={"book_flight": "cancel_flight"},
    alternative_map={"book_flight": ["book_flight_backup"]},
    retry_policy=RetryPolicy(max_retries=3, initial_delay=1.0),
    action_executor=your_framework_executor,  # Framework-specific
)
```

### 2. TransactionLog

Thread-safe log tracking compensatable actions with:

- Topological sorting for correct rollback order
- Multi-agent support via `agent_id`
- Serialization for persistence

```python
from react_agent_compensation.core import TransactionLog

log = TransactionLog()
record = manager.record_action("book_flight", {"dest": "NYC"})
# ... execute tool ...
manager.mark_completed(record.id, result={"booking_id": "123"})

# Get rollback plan (dependents first)
rollback_plan = log.get_rollback_plan(agent_id="agent1")
```

### 3. Protocol Interfaces

Framework-agnostic interfaces that adaptors must implement:

```python
from react_agent_compensation.core.protocols import ActionExecutor, ActionResult

class YourFrameworkExecutor(ActionExecutor):
    """Adapts your framework's tool execution to the core."""
    def execute(self, name: str, params: dict[str, Any]) -> ActionResult:
        tool = self._tools[name]
        result = tool.invoke(params)  # Framework-specific call
        return SimpleActionResult(content=result, name=name)
```

## Adaptor Methods

There are three primary methods for integrating the core with agent frameworks:

| **Method**                                                      | **Frameworks**                                                                                                | **How it works**                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Method A — Middleware / Filters / Callbacks (Interceptor)**   | Semantic Kernel (Python), LlamaIndex, Haystack, OpenAI Agents SDK, AutoGen, Griptape, LangChain/LangGraph     | Hook the framework's **pre/post tool invocation** lifecycle. Before execution: if tool is compensatable, call `record_action(tool, params)`. After success: `mark_completed(id, result)`. After failure/exception: `mark_failed(id, err)` then `recover(id, err)` (retry/backoff + alternatives). If recovery fails: `rollback()` which executes compensators via `ActionExecutor`. Persist `TransactionLog` in the framework's run/state context. |
| **Method B — Tool Wrapper / Decorator**                         | PydanticAI, smolagents, CrewAI, LangChain tools (direct wrapping), any "tools are Python callables" framework | Wrap each tool callable (`invoke/_run/func/async`) with a wrapper that runs the same sequence: `record_action → execute → mark_completed/failed → recover → rollback`. Build an `ActionExecutor` backed by the framework's tool registry (name → callable) so compensators can be invoked by name during rollback. Store `TransactionLog` on the agent/session object (or external store) for long-running runs.                                   |
| **Method C — External "Compensation Service" (out-of-process)** | Any framework (including non-Python), plus distributed multi-agent systems                                    | Run your core as a separate service. The framework adaptor sends tool-call events (tool name, params, result/error, run_id/agent_id) to the service. The service decides retry/alternative/rollback and returns actions to execute (including extracted compensation params). State (`TransactionLog`) lives centrally, enabling auditability, multi-runtime support, and shared rollback coordination.                                            |

## Method A: Middleware/Filters/Callbacks

This method intercepts tool execution at the framework's lifecycle hooks. It's the most "native" integration when the framework provides middleware/filter/callback mechanisms.

### Architecture Pattern

```
Framework Tool Call
    ↓
[Your Middleware/Filter]
    ├─→ record_action(tool, params)
    ├─→ Execute original tool
    ├─→ mark_completed(id, result) OR mark_failed(id, error)
    ├─→ recover(id, error) [retry + alternatives]
    └─→ rollback() [if recovery fails]
    ↓
Return result or error message to framework
```

### Example: Semantic Kernel Filter

```python
from semantic_kernel.functions import KernelFunction
from semantic_kernel.filters import FunctionInvokingFilter, FunctionInvokedFilter
from react_agent_compensation.core import RecoveryManager, SimpleActionResult
from react_agent_compensation.core.protocols import ActionExecutor

class SKActionExecutor(ActionExecutor):
    """ActionExecutor for Semantic Kernel functions."""
    def __init__(self, kernel):
        self._kernel = kernel

    def execute(self, name: str, params: dict[str, Any]) -> ActionResult:
        func = self._kernel.get_function_from_fully_qualified_name(name)
        result = func.invoke(kernel=self._kernel, **params)
        return SimpleActionResult(
            content=str(result.value),
            name=name,
            status="success" if not result.metadata.get("error") else "error"
        )

class CompensationFilter(FunctionInvokingFilter, FunctionInvokedFilter):
    """Semantic Kernel filter that adds compensation tracking."""

    def __init__(self, recovery_manager: RecoveryManager):
        self.rc_manager = recovery_manager
        self._current_record_id = None

    async def on_invoking(self, function: KernelFunction, arguments: dict) -> None:
        """Called before function execution."""
        tool_name = function.name
        params = {k: str(v) for k, v in arguments.items()}

        if self.rc_manager.is_compensatable(tool_name):
            record = self.rc_manager.record_action(tool_name, params)
            self._current_record_id = record.id

    async def on_invoked(self, function: KernelFunction, result: Any) -> None:
        """Called after function execution."""
        if self._current_record_id:
            # Check for errors
            if hasattr(result, 'metadata') and result.metadata.get('error'):
                error_msg = result.metadata.get('error', 'Unknown error')
                self.rc_manager.mark_failed(self._current_record_id, error_msg)

                # Try recovery
                recovery = self.rc_manager.recover(
                    self._current_record_id,
                    error_msg,
                    execute_fn=lambda name, args: self._execute_sk_function(name, args)
                )

                if not recovery.success:
                    # Trigger rollback
                    self.rc_manager.rollback(self._current_record_id)
            else:
                # Success
                self.rc_manager.mark_completed(
                    self._current_record_id,
                    str(result.value) if hasattr(result, 'value') else result
                )

            self._current_record_id = None

    def _execute_sk_function(self, name: str, params: dict) -> Any:
        """Helper to execute SK functions during recovery."""
        func = self.rc_manager._action_executor._kernel.get_function_from_fully_qualified_name(name)
        return func.invoke(kernel=self.rc_manager._action_executor._kernel, **params)

# Usage
from semantic_kernel import Kernel

kernel = Kernel()
executor = SKActionExecutor(kernel)
manager = RecoveryManager(
    compensation_pairs={"create_order": "cancel_order"},
    action_executor=executor,
)

# Add filter to kernel
kernel.add_filter(CompensationFilter(manager))
```

### Example: LlamaIndex Callback Handler

```python
from llama_index.core.callbacks import CallbackManager, CBEventType
from llama_index.core.callbacks.schema import EventPayload
from react_agent_compensation.core import RecoveryManager

class CompensationCallbackHandler:
    """LlamaIndex callback that adds compensation tracking."""

    def __init__(self, recovery_manager: RecoveryManager):
        self.rc_manager = recovery_manager
        self._current_record_id = None

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: dict | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Called when tool execution starts."""
        if event_type == CBEventType.FUNCTION_CALL:
            tool_name = payload.get(EventPayload.FUNCTION_NAME) if payload else None
            params = payload.get(EventPayload.FUNCTION_PARAMS, {}) if payload else {}

            if tool_name and self.rc_manager.is_compensatable(tool_name):
                record = self.rc_manager.record_action(tool_name, params)
                self._current_record_id = record.id

        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: dict | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Called when tool execution ends."""
        if event_type == CBEventType.FUNCTION_CALL and self._current_record_id:
            result = payload.get(EventPayload.FUNCTION_OUTPUT) if payload else None
            error = payload.get(EventPayload.EXCEPTION) if payload else None

            if error:
                self.rc_manager.mark_failed(self._current_record_id, str(error))
                recovery = self.rc_manager.recover(self._current_record_id, error)
                if not recovery.success:
                    self.rc_manager.rollback(self._current_record_id)
            else:
                self.rc_manager.mark_completed(self._current_record_id, result)

            self._current_record_id = None

# Usage
from llama_index.core.llms import LLM
from llama_index.core.agent import ReActAgent

manager = RecoveryManager(compensation_pairs={"book": "cancel"})
callback_handler = CompensationCallbackHandler(manager)

callback_manager = CallbackManager([callback_handler])
agent = ReActAgent.from_tools(
    tools=[...],
    llm=llm,
    callback_manager=callback_manager,
)
```

### Example: Haystack Pipeline Component

```python
from haystack import Pipeline, component
from haystack.components.agents import Agent
from react_agent_compensation.core import RecoveryManager

class CompensatedAgentComponent:
    """Haystack component wrapper that adds compensation."""

    def __init__(self, agent: Agent, recovery_manager: RecoveryManager):
        self.agent = agent
        self.rc_manager = recovery_manager

    @component.output_types(result=dict)
    def run(self, query: str, **kwargs):
        """Execute agent with compensation tracking."""
        # Execute agent (this internally calls tools)
        result = self.agent.run(query=query, **kwargs)

        # Check for failures and handle compensation
        # (Implementation depends on how Haystack reports tool failures)
        return {"result": result}

# Usage
pipeline = Pipeline()
manager = RecoveryManager(compensation_pairs={"create": "delete"})

compensated_agent = CompensatedAgentComponent(agent, manager)
pipeline.add_component("agent", compensated_agent)
```

## Method B: Tool Wrapper/Decorator

This method wraps individual tool callables. It's the most portable approach and works with any framework where tools are Python functions or callable objects.

### Architecture Pattern

```
Original Tool Function
    ↓
[Wrapper Function]
    ├─→ record_action(tool_name, params)
    ├─→ try: original_tool(*args, **kwargs)
    ├─→ mark_completed(id, result)
    ├─→ except: mark_failed(id, error)
    ├─→ recover(id, error) [retry + alternatives]
    └─→ rollback() [if recovery fails]
    ↓
Return result or formatted error message
```

### Example: PydanticAI Tool Wrapper

```python
from pydantic_ai import Agent
from pydantic_ai.tools import tool
from react_agent_compensation.core import RecoveryManager, SimpleActionResult
from react_agent_compensation.core.protocols import ActionExecutor
from functools import wraps

class PydanticAIActionExecutor(ActionExecutor):
    """ActionExecutor for PydanticAI tools."""
    def __init__(self, tools_registry: dict[str, callable]):
        self._tools = tools_registry

    def execute(self, name: str, params: dict[str, Any]) -> ActionResult:
        tool_func = self._tools.get(name)
        if not tool_func:
            return SimpleActionResult(
                content=f"Tool {name} not found",
                status="error",
                name=name
            )

        try:
            result = tool_func(**params)
            return SimpleActionResult(content=result, name=name, status="success")
        except Exception as e:
            return SimpleActionResult(
                content=str(e),
                status="error",
                name=name
            )

def wrap_pydantic_tool(
    tool_func: callable,
    recovery_manager: RecoveryManager,
    tool_name: str,
) -> callable:
    """Wrap a PydanticAI tool with compensation tracking."""

    @wraps(tool_func)
    async def wrapped_tool(*args, **kwargs):
        # Convert args to kwargs if needed
        params = kwargs.copy()
        if args:
            # Infer param names from tool signature
            import inspect
            sig = inspect.signature(tool_func)
            param_names = list(sig.parameters.keys())
            for i, arg in enumerate(args):
                if i < len(param_names):
                    params[param_names[i]] = arg

        record = None
        is_compensatable = recovery_manager.is_compensatable(tool_name)

        if is_compensatable:
            record = recovery_manager.record_action(tool_name, params)

        try:
            # Execute original tool
            result = await tool_func(*args, **kwargs) if asyncio.iscoroutinefunction(tool_func) else tool_func(*args, **kwargs)

            # Check for errors in result
            if isinstance(result, dict) and result.get("error"):
                error_msg = result["error"]
                if record:
                    recovery_manager.mark_failed(record.id, error_msg)
                    recovery = recovery_manager.recover(record.id, error_msg)
                    if not recovery.success:
                        recovery_manager.rollback(record.id)
                return {"error": error_msg}

            # Success
            if record:
                recovery_manager.mark_completed(record.id, result)

            return result

        except Exception as e:
            error_msg = str(e)
            if record:
                recovery_manager.mark_failed(record.id, error_msg)
                recovery = recovery_manager.recover(record.id, error_msg)
                if not recovery.success:
                    recovery_manager.rollback(record.id)
            raise

    return wrapped_tool

# Usage
manager = RecoveryManager(
    compensation_pairs={"book_flight": "cancel_flight"},
    action_executor=PydanticAIActionExecutor({}),
)

@tool
async def book_flight(destination: str, date: str) -> dict:
    """Book a flight."""
    # Your booking logic
    return {"booking_id": "123", "destination": destination}

@tool
async def cancel_flight(booking_id: str) -> dict:
    """Cancel a flight."""
    # Your cancellation logic
    return {"cancelled": True}

# Wrap tools
wrapped_book = wrap_pydantic_tool(book_flight, manager, "book_flight")
wrapped_cancel = wrap_pydantic_tool(cancel_flight, manager, "cancel_flight")

# Register with agent
agent = Agent(
    'gpt-4',
    tools=[wrapped_book, wrapped_cancel],
)
```

### Example: smolagents Tool Wrapper

```python
from smolagents import Tool, CodeAgent
from react_agent_compensation.core import RecoveryManager

def wrap_smol_tool(
    tool: Tool,
    recovery_manager: RecoveryManager,
) -> Tool:
    """Wrap a smolagents Tool with compensation tracking."""

    original_call = tool.call

    def wrapped_call(*args, **kwargs):
        tool_name = tool.name
        params = kwargs.copy()
        if args:
            # Map positional args if tool has signature
            import inspect
            sig = inspect.signature(original_call)
            param_names = list(sig.parameters.keys())
            for i, arg in enumerate(args):
                if i < len(param_names):
                    params[param_names[i]] = arg

        record = None
        if recovery_manager.is_compensatable(tool_name):
            record = recovery_manager.record_action(tool_name, params)

        try:
            result = original_call(*args, **kwargs)

            if record:
                if isinstance(result, dict) and result.get("error"):
                    recovery_manager.mark_failed(record.id, result["error"])
                    recovery = recovery_manager.recover(record.id, result["error"])
                    if not recovery.success:
                        recovery_manager.rollback(record.id)
                else:
                    recovery_manager.mark_completed(record.id, result)

            return result

        except Exception as e:
            if record:
                recovery_manager.mark_failed(record.id, str(e))
                recovery = recovery_manager.recover(record.id, e)
                if not recovery.success:
                    recovery_manager.rollback(record.id)
            raise

    # Create new Tool with wrapped call
    wrapped_tool = Tool(
        name=tool.name,
        description=tool.description,
        call=wrapped_call,
    )

    return wrapped_tool

# Usage
manager = RecoveryManager(compensation_pairs={"create": "delete"})

original_tool = Tool(
    name="create_item",
    description="Create an item",
    call=lambda name: {"id": "123", "name": name},
)

wrapped_tool = wrap_smol_tool(original_tool, manager)

agent = CodeAgent(
    tools=[wrapped_tool],
    model="gpt-4",
)
```

### Example: CrewAI Tool Wrapper

```python
from crewai.tools import tool
from crewai import Agent, Crew, Task
from react_agent_compensation.core import RecoveryManager
from functools import wraps

def wrap_crew_tool(
    tool_func: callable,
    recovery_manager: RecoveryManager,
    tool_name: str,
) -> callable:
    """Wrap a CrewAI tool with compensation tracking."""

    @wraps(tool_func)
    def wrapped_tool(*args, **kwargs):
        params = kwargs.copy()
        if args:
            import inspect
            sig = inspect.signature(tool_func)
            param_names = list(sig.parameters.keys())
            for i, arg in enumerate(args):
                if i < len(param_names):
                    params[param_names[i]] = arg

        record = None
        if recovery_manager.is_compensatable(tool_name):
            record = recovery_manager.record_action(tool_name, params)

        try:
            result = tool_func(*args, **kwargs)

            if record:
                if isinstance(result, dict) and result.get("error"):
                    recovery_manager.mark_failed(record.id, result["error"])
                    recovery = recovery_manager.recover(record.id, result["error"])
                    if not recovery.success:
                        recovery_manager.rollback(record.id)
                else:
                    recovery_manager.mark_completed(record.id, result)

            return result

        except Exception as e:
            if record:
                recovery_manager.mark_failed(record.id, str(e))
                recovery = recovery_manager.recover(record.id, e)
                if not recovery.success:
                    recovery_manager.rollback(record.id)
            raise

    return wrapped_tool

# Usage
manager = RecoveryManager(compensation_pairs={"book": "cancel"})

@tool
def book_hotel(city: str, check_in: str) -> str:
    """Book a hotel."""
    return f"Booked hotel in {city}"

@tool
def cancel_hotel(booking_id: str) -> str:
    """Cancel a hotel booking."""
    return f"Cancelled booking {booking_id}"

# Wrap tools
wrapped_book = wrap_crew_tool(book_hotel, manager, "book_hotel")
wrapped_cancel = wrap_crew_tool(cancel_hotel, manager, "cancel_hotel")

agent = Agent(
    role="Travel Agent",
    goal="Book travel arrangements",
    tools=[wrapped_book, wrapped_cancel],
)
```

## Method C: External Compensation Service

This method runs the compensation core as a separate service, enabling cross-language support, centralized auditability, and distributed multi-agent coordination.

### Architecture Pattern

```
Framework Agent
    ↓
[Adaptor Client]
    ├─→ Send tool call event to service
    │   POST /compensation/record
    │   {tool_name, params, run_id, agent_id}
    │
    ├─→ Execute tool
    │
    ├─→ Send result/error to service
    │   POST /compensation/complete
    │   {record_id, result/error}
    │
    └─→ Service responds with:
        - retry? (with delay)
        - try alternative?
        - rollback? (with compensation params)
    ↓
Return result or service-provided error message
```

### Example: HTTP Service Implementation

```python
# service.py - Compensation Service
from fastapi import FastAPI, HTTPException
from react_agent_compensation.core import RecoveryManager, TransactionLog
from pydantic import BaseModel

app = FastAPI()

# Shared recovery managers per run_id
managers: dict[str, RecoveryManager] = {}

class RecordActionRequest(BaseModel):
    tool_name: str
    params: dict[str, Any]
    run_id: str
    agent_id: str | None = None

class CompleteActionRequest(BaseModel):
    record_id: str
    result: Any | None = None
    error: str | None = None

@app.post("/compensation/record")
async def record_action(request: RecordActionRequest):
    """Record a tool action before execution."""
    if request.run_id not in managers:
        managers[request.run_id] = RecoveryManager(
            compensation_pairs={},  # Load from config/DB
            shared_log=TransactionLog(),
        )

    manager = managers[request.run_id]
    record = manager.record_action(
        request.tool_name,
        request.params,
    )

    return {
        "record_id": record.id,
        "compensatable": manager.is_compensatable(request.tool_name),
    }

@app.post("/compensation/complete")
async def complete_action(request: CompleteActionRequest):
    """Mark action as completed or failed."""
    # Find manager by record_id (would need to store mapping)
    # For simplicity, assume we can look it up

    manager = find_manager_for_record(request.record_id)

    if request.error:
        manager.mark_failed(request.record_id, request.error)
        recovery = manager.recover(request.record_id, request.error)

        if recovery.success:
            return {
                "action": "retry" if recovery.action_taken == "retry" else "alternative",
                "result": recovery.result,
                "attempts": recovery.attempts,
            }
        else:
            # Trigger rollback
            rollback_result = manager.rollback(request.record_id)
            return {
                "action": "rollback",
                "compensated": rollback_result.compensated,
                "message": rollback_result.message,
            }
    else:
        manager.mark_completed(request.record_id, request.result)
        return {"action": "completed"}

@app.get("/compensation/rollback-plan/{run_id}")
async def get_rollback_plan(run_id: str, agent_id: str | None = None):
    """Get rollback plan for a run."""
    if run_id not in managers:
        return {"plan": []}

    manager = managers[run_id]
    plan = manager.log.get_rollback_plan(agent_id=agent_id)

    return {
        "plan": [
            {
                "id": r.id,
                "action": r.action,
                "compensator": r.compensator,
                "params": r.params,
            }
            for r in plan
        ]
    }
```

### Example: Framework Adaptor Client

```python
# adaptor_client.py - Framework-side client
import httpx
from typing import Any

class CompensationServiceClient:
    """Client for communicating with compensation service."""

    def __init__(self, service_url: str):
        self.service_url = service_url
        self.client = httpx.AsyncClient(base_url=service_url)
        self._record_ids: dict[str, str] = {}  # tool_call_id -> record_id

    async def record_action(
        self,
        tool_name: str,
        params: dict[str, Any],
        run_id: str,
        tool_call_id: str,
        agent_id: str | None = None,
    ) -> bool:
        """Record action before execution."""
        response = await self.client.post(
            "/compensation/record",
            json={
                "tool_name": tool_name,
                "params": params,
                "run_id": run_id,
                "agent_id": agent_id,
            }
        )
        response.raise_for_status()
        data = response.json()

        if data["compensatable"]:
            self._record_ids[tool_call_id] = data["record_id"]
            return True
        return False

    async def complete_action(
        self,
        tool_call_id: str,
        result: Any | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Mark action complete and get recovery instructions."""
        record_id = self._record_ids.get(tool_call_id)
        if not record_id:
            return {"action": "none"}

        response = await self.client.post(
            "/compensation/complete",
            json={
                "record_id": record_id,
                "result": result,
                "error": error,
            }
        )
        response.raise_for_status()
        return response.json()

# Usage in framework adaptor
class FrameworkCompensationAdaptor:
    def __init__(self, service_url: str, run_id: str):
        self.client = CompensationServiceClient(service_url)
        self.run_id = run_id

    async def wrap_tool_call(self, tool_name: str, params: dict, tool_call_id: str):
        """Wrap tool execution with compensation service."""
        # Record action
        compensatable = await self.client.record_action(
            tool_name, params, self.run_id, tool_call_id
        )

        # Execute tool
        try:
            result = await execute_tool(tool_name, params)
            response = await self.client.complete_action(tool_call_id, result=result)

            if response["action"] == "completed":
                return result
            elif response["action"] in ("retry", "alternative"):
                # Service says to use alternative result
                return response["result"]
            else:
                # Rollback occurred
                return {
                    "error": response["message"],
                    "rolled_back": True,
                }

        except Exception as e:
            response = await self.client.complete_action(
                tool_call_id, error=str(e)
            )

            if response["action"] == "rollback":
                return {
                    "error": response["message"],
                    "rolled_back": True,
                }
            raise
```

## Framework-Specific Examples

### LangChain/LangGraph (Reference Implementation)

The existing LangChain adaptor demonstrates both Method A (middleware) and Method B (tool wrapper):

**Method A - Middleware:**

```python
from react_agent_compensation.langchain_adaptor import CompensationMiddleware

middleware = CompensationMiddleware(
    compensation_mapping={"book": "cancel"},
    tools=tools,
)

agent = create_react_agent(model, tools=tools, middleware=[middleware])
```

**Method B - Tool Wrapper:**

```python
from react_agent_compensation.langchain_adaptor import create_compensated_agent

agent = create_compensated_agent(
    model=model,
    tools=tools,
    compensation_mapping={"book": "cancel"},
)
```

See `src/react_agent_compensation/langchain_adaptor/` for the full implementation.

### OpenAI Agents SDK

```python
from openai import OpenAI
from openai.agents import Agent, Runner
from react_agent_compensation.core import RecoveryManager

# Create executor for OpenAI Agents SDK
class OpenAIAgentsExecutor(ActionExecutor):
    def __init__(self, runner: Runner):
        self.runner = runner

    def execute(self, name: str, params: dict[str, Any]) -> ActionResult:
        # Execute tool via runner
        result = self.runner.call_tool(name, params)
        return SimpleActionResult(content=result, name=name)

# Hook into runner's tool execution
manager = RecoveryManager(
    compensation_pairs={"create": "delete"},
    action_executor=OpenAIAgentsExecutor(runner),
)

# Wrap runner's tool execution
original_call_tool = runner.call_tool

def compensated_call_tool(tool_name: str, params: dict):
    record = None
    if manager.is_compensatable(tool_name):
        record = manager.record_action(tool_name, params)

    try:
        result = original_call_tool(tool_name, params)
        if record:
            manager.mark_completed(record.id, result)
        return result
    except Exception as e:
        if record:
            manager.mark_failed(record.id, str(e))
            recovery = manager.recover(record.id, e)
            if not recovery.success:
                manager.rollback(record.id)
        raise

runner.call_tool = compensated_call_tool
```

### AutoGen

```python
from autogen import ConversableAgent
from react_agent_compensation.core import RecoveryManager

# Shared log for multi-agent coordination
shared_log = TransactionLog()

manager = RecoveryManager(
    compensation_pairs={"book": "cancel"},
    shared_log=shared_log,
    agent_id="travel_agent",
)

# Wrap function registration
def register_compensated_function(agent: ConversableAgent, func, name: str):
    """Register function with compensation tracking."""

    def wrapped_func(**kwargs):
        record = None
        if manager.is_compensatable(name):
            record = manager.record_action(name, kwargs)

        try:
            result = func(**kwargs)
            if record:
                manager.mark_completed(record.id, result)
            return result
        except Exception as e:
            if record:
                manager.mark_failed(record.id, str(e))
                recovery = manager.recover(record.id, e)
                if not recovery.success:
                    manager.rollback(record.id)
            raise

    agent.register_for_execution(name)(wrapped_func)
    agent.register_for_llm(name, description=func.__doc__)

# Usage
agent = ConversableAgent(
    name="travel_agent",
    system_message="You are a travel agent.",
)

def book_flight(dest: str) -> str:
    return f"Booked flight to {dest}"

def cancel_flight(booking_id: str) -> str:
    return f"Cancelled {booking_id}"

register_compensated_function(agent, book_flight, "book_flight")
register_compensated_function(agent, cancel_flight, "cancel_flight")
```

## Best Practices

### 1. State Persistence

For long-running agents or multi-step workflows, persist the `TransactionLog`:

```python
# Save to framework state
state["compensation_log"] = manager.log.to_dict()

# Restore from framework state
log_data = state.get("compensation_log", {})
manager._log = TransactionLog.from_dict(log_data)
```

### 2. Multi-Agent Coordination

Use shared `TransactionLog` for coordinated rollback:

```python
shared_log = TransactionLog()

manager1 = RecoveryManager(
    compensation_pairs={...},
    shared_log=shared_log,
    agent_id="agent1",
)

manager2 = RecoveryManager(
    compensation_pairs={...},
    shared_log=shared_log,
    agent_id="agent2",
)

# Rollback can filter by agent_id
rollback_plan = shared_log.get_rollback_plan(agent_id="agent1")
```

### 3. Error Detection

Customize error detection for your framework's result format:

```python
from react_agent_compensation.core.errors import create_error_detector

# Custom error detector
class MyFrameworkErrorDetector(ErrorStrategy):
    def is_error(self, result: Any) -> bool | None:
        # Framework-specific error detection
        if isinstance(result, MyFrameworkResult):
            return result.status == "error"
        return None

error_detector = create_error_detector([MyFrameworkErrorDetector()])
```

### 4. Parameter Extraction

Configure extraction strategies for compensation parameters:

```python
from react_agent_compensation.core.extraction import CompensationSchema

schemas = {
    "book_flight": CompensationSchema(
        param_mapping={"booking_id": "result.id"},
        static_params={"reason": "Auto rollback"},
    )
}

manager = RecoveryManager(
    compensation_pairs={"book_flight": "cancel_flight"},
    compensation_schemas=schemas,
)
```

### 5. Testing Adaptors

Test adaptors with the same patterns as the LangChain tests:

```python
def test_adaptor_records_action():
    manager = RecoveryManager(compensation_pairs={"book": "cancel"})
    adaptor = YourFrameworkAdaptor(manager)

    record = manager.record_action("book", {"dest": "NYC"})
    result = adaptor.execute_tool("book", {"dest": "NYC"})
    manager.mark_completed(record.id, result)

    assert len(manager.log) == 1
    assert manager.log.get(record.id).status == ActionStatus.COMPLETED
```

## Summary

The `react-agent-compensation` core is designed to be framework-agnostic through:

1. **Protocol-based interfaces** (`ActionExecutor`, `ActionResult`) that adaptors implement
2. **Pluggable strategies** for extraction, error detection, and retry
3. **Serializable state** (`TransactionLog`) for persistence
4. **Multi-agent support** via `agent_id` and shared logs

Choose the adaptor method based on your framework:

- **Method A** for frameworks with middleware/filters/callbacks
- **Method B** for frameworks where tools are callables
- **Method C** for distributed systems or cross-language support

The LangChain adaptor (`src/react_agent_compensation/langchain_adaptor/`) serves as a reference implementation demonstrating both Method A and Method B patterns.
