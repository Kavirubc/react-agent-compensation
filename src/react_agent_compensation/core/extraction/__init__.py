"""Extraction strategies for compensation parameter extraction.

This module provides pluggable strategies for extracting parameters
needed to execute compensation tools from tool results.

Strategies (in default priority order):
1. StateMappersStrategy - Developer-provided functions
2. SchemaExtractionStrategy - Declarative path expressions
3. HeuristicExtractionStrategy - Common ID field detection
4. RecursiveSearchStrategy - Deep nested structure search
5. PassthroughStrategy - Pass entire result through

Factory function:
- create_extraction_strategy() - Create configured strategy chain
"""

from react_agent_compensation.core.extraction.base import (
    CompensationSchema,
    ExtractionStrategy,
    ToolLike,
)
from react_agent_compensation.core.extraction.composite import (
    CompositeExtractionStrategy,
    create_extraction_strategy,
)
from react_agent_compensation.core.extraction.heuristic import HeuristicExtractionStrategy
from react_agent_compensation.core.extraction.passthrough import PassthroughStrategy
from react_agent_compensation.core.extraction.path_resolver import (
    extract_all_values,
    resolve_path,
    validate_path,
)
from react_agent_compensation.core.extraction.recursive import RecursiveSearchStrategy
from react_agent_compensation.core.extraction.schema import SchemaExtractionStrategy
from react_agent_compensation.core.extraction.state_mappers import (
    StateMapperFn,
    StateMappersStrategy,
)

__all__ = [
    # Base classes
    "ExtractionStrategy",
    "CompensationSchema",
    "ToolLike",
    # Concrete strategies
    "StateMappersStrategy",
    "StateMapperFn",
    "SchemaExtractionStrategy",
    "HeuristicExtractionStrategy",
    "RecursiveSearchStrategy",
    "PassthroughStrategy",
    "CompositeExtractionStrategy",
    # Factory
    "create_extraction_strategy",
    # Path utilities
    "resolve_path",
    "validate_path",
    "extract_all_values",
]
