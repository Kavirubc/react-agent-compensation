"""LLM-based extraction strategy with multi-provider support.

Uses large language models to intelligently extract compensation
parameters when heuristic strategies fail.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from react_agent_compensation.core.extraction.base import ExtractionStrategy, ToolLike


logger = logging.getLogger(__name__)


class LLMExtractionStrategy(ExtractionStrategy):
    """Use LLM to extract compensation parameters when heuristics fail.

    This strategy uses an LLM to intelligently extract compensation
    parameters from complex tool results when simpler strategies cannot
    determine the correct mapping.

    Features:
    - Lazy initialization (LLM client only created when needed)
    - Multi-provider support (OpenAI, Anthropic)
    - Result caching to avoid repeated LLM calls
    - Configurable prompt template

    Example:
        strategy = LLMExtractionStrategy(model="gpt-4o-mini")

        # Or with pre-configured client
        from openai import OpenAI
        strategy = LLMExtractionStrategy(client=OpenAI())
    """

    DEFAULT_PROMPT = '''You are a parameter extraction assistant. Extract parameters needed to call a compensation (rollback) tool.

Original Tool: {tool_name}
Original Parameters: {original_params}
Tool Result: {result}

Compensation Tool: {comp_tool_name}
Expected Parameters: {comp_schema}

Extract the parameters needed to call the compensation tool.
Return ONLY a valid JSON object with the extracted parameters.
Do not include any explanation or markdown formatting.

Example: {{"booking_id": "ABC123", "reason": "automatic rollback"}}

Your response:'''

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client: Any | None = None,
        provider: str | None = None,
        cache_extractions: bool = True,
        max_retries: int = 2,
        temperature: float = 0.0,
        prompt_template: str | None = None,
    ):
        """Initialize LLM extraction strategy.

        Args:
            model: Model identifier (e.g., "gpt-4o-mini", "claude-3-haiku")
            client: Pre-configured LLM client. If None, lazily created.
            provider: Provider name ("openai", "anthropic"). Auto-detected if None.
            cache_extractions: Cache results to avoid repeated LLM calls.
            max_retries: Number of retries on parse failure.
            temperature: LLM temperature (0.0 for deterministic).
            prompt_template: Custom prompt template (uses DEFAULT_PROMPT if None).
        """
        self._model = model
        self._client = client
        self._provider = provider or self._detect_provider(model)
        self.cache_extractions = cache_extractions
        self.max_retries = max_retries
        self.temperature = temperature
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self._cache: dict[str, dict[str, Any]] = {}

    def _detect_provider(self, model: str) -> str:
        """Detect provider from model name."""
        model_lower = model.lower()
        if any(name in model_lower for name in ["gpt", "o1", "text-"]):
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        return "openai"  # Default

    def _get_client(self) -> Any:
        """Lazy-load the LLM client."""
        if self._client is not None:
            return self._client

        if self._provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError as e:
                raise ImportError(
                    "OpenAI package required. Install with: pip install openai"
                ) from e
        elif self._provider == "anthropic":
            try:
                from anthropic import Anthropic
                self._client = Anthropic()
            except ImportError as e:
                raise ImportError(
                    "Anthropic package required. Install with: pip install anthropic"
                ) from e
        else:
            raise ValueError(f"Unsupported provider: {self._provider}")

        return self._client

    def _get_tool_schema(self, tool: ToolLike | None) -> str:
        """Extract schema description from a tool."""
        if tool is None:
            return "Unknown - extract common ID fields like 'id', 'booking_id'"

        try:
            if hasattr(tool, "get_input_schema"):
                schema = tool.get_input_schema()
                if "properties" in schema:
                    parts = []
                    for name, prop in schema["properties"].items():
                        prop_type = prop.get("type", "any")
                        desc = prop.get("description", "")
                        parts.append(f"- {name} ({prop_type}): {desc}")
                    return "\n".join(parts)
        except Exception:
            pass

        return "Unknown schema"

    def _cache_key(self, tool_name: str, result: Any) -> str:
        """Generate cache key."""
        result_str = json.dumps(result, sort_keys=True, default=str)
        return hashlib.md5(f"{tool_name}:{result_str}".encode()).hexdigest()

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return response text."""
        client = self._get_client()

        if self._provider == "openai":
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return response.choices[0].message.content or ""

        elif self._provider == "anthropic":
            response = client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text if response.content else ""

        raise ValueError(f"Unsupported provider: {self._provider}")

    def _parse_response(self, response: str) -> dict[str, Any] | None:
        """Parse LLM response to extract JSON."""
        response = response.strip()

        # Remove markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            response = "\n".join(lines).strip()

        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Find JSON object in response
        match = re.search(r"\{[^{}]*\}", response)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try to find nested JSON
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None

    def extract(
        self,
        result: Any,
        original_params: dict[str, Any],
        compensation_tool: ToolLike | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Extract compensation parameters using LLM.

        Args:
            result: The result from the original tool call
            original_params: The original parameters passed to the tool
            compensation_tool: The compensation tool (for schema inspection)
            tool_name: Name of the original tool

        Returns:
            Extracted parameters dict, or None if extraction failed
        """
        # Check cache
        if self.cache_extractions and tool_name:
            cache_key = self._cache_key(tool_name, result)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for {tool_name}")
                return self._cache[cache_key]

        # Build prompt
        prompt = self.prompt_template.format(
            tool_name=tool_name or "unknown",
            original_params=json.dumps(original_params, default=str),
            result=json.dumps(result, default=str) if not isinstance(result, str) else result,
            comp_tool_name=compensation_tool.name if compensation_tool else "unknown",
            comp_schema=self._get_tool_schema(compensation_tool),
        )

        # Call LLM with retries
        for attempt in range(self.max_retries + 1):
            try:
                response = self._call_llm(prompt)
                extracted = self._parse_response(response)

                if extracted and isinstance(extracted, dict):
                    # Cache result
                    if self.cache_extractions and tool_name:
                        cache_key = self._cache_key(tool_name, result)
                        self._cache[cache_key] = extracted
                    logger.debug(f"LLM extracted params for {tool_name}: {extracted}")
                    return extracted
            except Exception as e:
                logger.warning(f"LLM extraction attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries:
                    return None
                continue

        return None

    @property
    def name(self) -> str:
        """Return strategy name with model."""
        return f"LLMExtractionStrategy({self._model})"

    def clear_cache(self) -> None:
        """Clear the extraction cache."""
        self._cache.clear()
