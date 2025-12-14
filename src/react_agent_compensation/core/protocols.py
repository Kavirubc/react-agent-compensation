"""Protocol definitions for framework-agnostic interfaces.

This module defines Protocol classes that allow the Core module to work
with any framework (LangChain, custom, etc.) by defining abstract interfaces.

Protocols:
- ActionResult: Interface for tool execution results
- ActionExecutor: Interface for executing tools
- ToolSchemaProvider: Interface for tools that provide JSON schemas
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ActionResult(Protocol):
    """Protocol for action/tool execution results.

    Implement this protocol to adapt framework-specific result types
    (e.g., LangChain's ToolMessage) to work with the compensation system.
    """

    @property
    def content(self) -> Any:
        """The content/payload of the result."""
        ...

    @property
    def status(self) -> str | None:
        """Optional status indicator (e.g., 'error', 'success')."""
        ...

    @property
    def name(self) -> str:
        """Name of the tool that produced this result."""
        ...


@runtime_checkable
class ActionExecutor(Protocol):
    """Protocol for executing actions/tools.

    Implement this protocol to provide a way to execute tools by name.
    Used by RecoveryManager to execute compensation tools.
    """

    def execute(self, name: str, params: dict[str, Any]) -> ActionResult:
        """Execute a tool/action by name with given parameters.

        Args:
            name: Name of the tool to execute
            params: Parameters to pass to the tool

        Returns:
            ActionResult containing the execution result
        """
        ...


@runtime_checkable
class ToolSchemaProvider(Protocol):
    """Protocol for tools that provide JSON Schema definitions.

    Implement this protocol to allow MCP integration to discover
    compensation pairs from tool schemas.
    """

    @property
    def name(self) -> str:
        """Name of the tool."""
        ...

    @property
    def description(self) -> str:
        """Description of what the tool does."""
        ...

    def get_input_schema(self) -> dict[str, Any]:
        """Get JSON Schema for the tool's input parameters.

        Returns:
            JSON Schema dict describing expected parameters
        """
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM invocation.

    Implement this protocol to provide LLM capabilities for
    intelligent parameter extraction.
    """

    def invoke(self, prompt: str) -> str:
        """Invoke the LLM with a prompt and return the response.

        Args:
            prompt: The prompt text to send to the LLM

        Returns:
            The LLM's response as a string
        """
        ...


class SimpleActionResult:
    """Simple implementation of ActionResult for internal use."""

    def __init__(
        self,
        content: Any,
        status: str | None = None,
        name: str = "",
    ):
        self._content = content
        self._status = status
        self._name = name

    @property
    def content(self) -> Any:
        return self._content

    @property
    def status(self) -> str | None:
        return self._status

    @property
    def name(self) -> str:
        return self._name
