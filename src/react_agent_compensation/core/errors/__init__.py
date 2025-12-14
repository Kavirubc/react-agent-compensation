"""Error detection strategies for compensation operations.

This module provides pluggable strategies for detecting errors
in tool execution results:
- ExplicitStatusStrategy - Check status attributes
- ContentDictStrategy - Check content dict patterns
- ExceptionContentStrategy - Check exception string patterns
- CompositeErrorStrategy - Chain multiple strategies

Factory function:
- create_error_detector() - Create configured detector chain
"""

from react_agent_compensation.core.errors.base import ErrorStrategy
from react_agent_compensation.core.errors.composite import (
    AlwaysErrorStrategy,
    CompositeErrorStrategy,
    NeverErrorStrategy,
    create_error_detector,
)
from react_agent_compensation.core.errors.content import ContentDictStrategy
from react_agent_compensation.core.errors.exception import ExceptionContentStrategy
from react_agent_compensation.core.errors.explicit import ExplicitStatusStrategy

__all__ = [
    # Base
    "ErrorStrategy",
    # Concrete strategies
    "ExplicitStatusStrategy",
    "ContentDictStrategy",
    "ExceptionContentStrategy",
    "CompositeErrorStrategy",
    # Testing strategies
    "AlwaysErrorStrategy",
    "NeverErrorStrategy",
    # Factory
    "create_error_detector",
]
