"""Tests for retry strategies."""

import pytest

from react_agent_compensation.core.config import RetryPolicy
from react_agent_compensation.core.retry import (
    ExponentialBackoffStrategy,
    FixedDelayStrategy,
    LinearBackoffStrategy,
    NoRetryStrategy,
    RetryContext,
)


class TestRetryContext:
    """Tests for RetryContext."""

    def test_create_context(self):
        """Test creating a retry context."""
        ctx = RetryContext(
            error=ValueError("test error"),
            attempt=1,
            action="test_tool",
            params={"key": "value"},
        )

        assert ctx.attempt == 1
        assert ctx.action == "test_tool"
        assert ctx.params == {"key": "value"}
        assert isinstance(ctx.error, ValueError)

    def test_default_values(self):
        """Test default context values."""
        ctx = RetryContext(error="error", attempt=1)

        assert ctx.action == ""
        assert ctx.params == {}
        assert ctx.elapsed_time == 0.0


class TestNoRetryStrategy:
    """Tests for NoRetryStrategy."""

    def test_never_retries(self):
        """Test that it never retries."""
        strategy = NoRetryStrategy()
        ctx = RetryContext(error="error", attempt=1)

        assert strategy.should_retry(ctx) is False
        assert strategy.should_retry(RetryContext(error="error", attempt=0)) is False

    def test_zero_delay(self):
        """Test that delay is always zero."""
        strategy = NoRetryStrategy()
        ctx = RetryContext(error="error", attempt=1)

        assert strategy.get_delay(ctx) == 0.0

    def test_name(self):
        """Test strategy name."""
        strategy = NoRetryStrategy()
        assert strategy.name == "NoRetryStrategy"


class TestFixedDelayStrategy:
    """Tests for FixedDelayStrategy."""

    def test_retries_within_limit(self):
        """Test retries up to max_retries."""
        strategy = FixedDelayStrategy(max_retries=3, delay=1.0)

        assert strategy.should_retry(RetryContext(error="e", attempt=1)) is True
        assert strategy.should_retry(RetryContext(error="e", attempt=2)) is True
        assert strategy.should_retry(RetryContext(error="e", attempt=3)) is False

    def test_fixed_delay(self):
        """Test that delay is always fixed."""
        strategy = FixedDelayStrategy(max_retries=3, delay=5.0)

        assert strategy.get_delay(RetryContext(error="e", attempt=1)) == 5.0
        assert strategy.get_delay(RetryContext(error="e", attempt=2)) == 5.0
        assert strategy.get_delay(RetryContext(error="e", attempt=3)) == 5.0


class TestLinearBackoffStrategy:
    """Tests for LinearBackoffStrategy."""

    def test_retries_within_limit(self):
        """Test retries up to max_retries."""
        strategy = LinearBackoffStrategy(max_retries=3, jitter=False)

        assert strategy.should_retry(RetryContext(error="e", attempt=1)) is True
        assert strategy.should_retry(RetryContext(error="e", attempt=2)) is True
        assert strategy.should_retry(RetryContext(error="e", attempt=3)) is False

    def test_linear_delay(self):
        """Test that delay increases linearly."""
        strategy = LinearBackoffStrategy(
            max_retries=5, delay_increment=2.0, jitter=False
        )

        assert strategy.get_delay(RetryContext(error="e", attempt=1)) == 2.0
        assert strategy.get_delay(RetryContext(error="e", attempt=2)) == 4.0
        assert strategy.get_delay(RetryContext(error="e", attempt=3)) == 6.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        strategy = LinearBackoffStrategy(
            max_retries=10, delay_increment=10.0, max_delay=25.0, jitter=False
        )

        assert strategy.get_delay(RetryContext(error="e", attempt=3)) == 25.0
        assert strategy.get_delay(RetryContext(error="e", attempt=5)) == 25.0

    def test_jitter(self):
        """Test that jitter adds randomness."""
        strategy = LinearBackoffStrategy(
            max_retries=3, delay_increment=10.0, jitter=True
        )
        ctx = RetryContext(error="e", attempt=1)

        # With jitter, delay should vary around 10.0
        delays = [strategy.get_delay(ctx) for _ in range(10)]
        assert not all(d == delays[0] for d in delays)  # Not all same
        assert all(8.0 <= d <= 12.0 for d in delays)  # Within range


class TestExponentialBackoffStrategy:
    """Tests for ExponentialBackoffStrategy."""

    def test_retries_within_limit(self):
        """Test retries up to max_retries."""
        policy = RetryPolicy(max_retries=3)
        strategy = ExponentialBackoffStrategy(policy)

        # Error must contain retryable pattern like "connection"
        assert strategy.should_retry(RetryContext(error="connection failed", attempt=1)) is True
        assert strategy.should_retry(RetryContext(error="connection failed", attempt=2)) is True
        assert strategy.should_retry(RetryContext(error="connection failed", attempt=3)) is False

    def test_exponential_delay(self):
        """Test that delay increases exponentially."""
        policy = RetryPolicy(
            max_retries=5,
            initial_delay=1.0,
            backoff_multiplier=2.0,
            jitter=False,
        )
        strategy = ExponentialBackoffStrategy(policy)

        assert strategy.get_delay(RetryContext(error="e", attempt=1)) == 1.0
        assert strategy.get_delay(RetryContext(error="e", attempt=2)) == 2.0
        assert strategy.get_delay(RetryContext(error="e", attempt=3)) == 4.0
        assert strategy.get_delay(RetryContext(error="e", attempt=4)) == 8.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        policy = RetryPolicy(
            max_retries=10,
            initial_delay=1.0,
            backoff_multiplier=10.0,
            max_delay=50.0,
            jitter=False,
        )
        strategy = ExponentialBackoffStrategy(policy)

        assert strategy.get_delay(RetryContext(error="e", attempt=3)) == 50.0

    def test_jitter(self):
        """Test that jitter adds randomness."""
        policy = RetryPolicy(
            max_retries=3, initial_delay=10.0, jitter=True
        )
        strategy = ExponentialBackoffStrategy(policy)
        ctx = RetryContext(error="e", attempt=1)

        delays = [strategy.get_delay(ctx) for _ in range(10)]
        assert not all(d == delays[0] for d in delays)  # Not all same
        assert all(5.0 <= d <= 15.0 for d in delays)  # Within jitter range

    def test_retryable_errors(self):
        """Test retryable error filtering."""
        policy = RetryPolicy(
            max_retries=3,
            retryable_errors=["connection", "timeout"],  # String patterns
        )
        strategy = ExponentialBackoffStrategy(policy)

        # Retryable error - contains "connection"
        ctx = RetryContext(error=ConnectionError("connection refused"), attempt=1)
        assert strategy.should_retry(ctx) is True

        # Non-retryable error - doesn't match patterns
        ctx = RetryContext(error=ValueError("bad input"), attempt=1)
        assert strategy.should_retry(ctx) is False

    def test_name(self):
        """Test strategy name includes config."""
        policy = RetryPolicy(max_retries=5)
        strategy = ExponentialBackoffStrategy(policy)
        assert "5" in strategy.name

    def test_default_policy(self):
        """Test default policy is used when None."""
        strategy = ExponentialBackoffStrategy(None)
        assert strategy.policy is not None
        assert strategy.policy.max_retries == 3  # Default

