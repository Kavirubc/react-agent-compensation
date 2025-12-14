"""LLM-based extraction strategies for compensation parameters.

This module provides intelligent extraction using large language models
when heuristic strategies fail. Requires optional dependencies:
- pip install openai (for OpenAI models)
- pip install anthropic (for Anthropic models)

Example:
    from react_agent_compensation.llm import LLMExtractionStrategy

    strategy = LLMExtractionStrategy(model="gpt-4o-mini")
    params = strategy.extract(result, original_params, tool_name="book_flight")
"""

from react_agent_compensation.llm.extraction import LLMExtractionStrategy

__all__ = [
    "LLMExtractionStrategy",
]
