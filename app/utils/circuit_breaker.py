"""Circuit breaker pattern implementation.

This module provides a circuit breaker to prevent cascade failures
when external services are unavailable or experiencing issues.
"""

import asyncio
import functools
import logging
import time
from enum import Enum
from typing import Any, Callable

from app.core.config import settings
from app.core.exceptions import CircuitBreakerOpenError

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit is open, requests are blocked
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreaker:
    """Circuit breaker to prevent cascade failures.

    The circuit breaker monitors failures and opens the circuit when
    a failure threshold is reached. After a timeout, it enters a half-open
    state to test if the service has recovered.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        timeout: Time in seconds before attempting recovery
        half_open_timeout: Time in seconds for half-open state
        state: Current circuit state
        failure_count: Current count of consecutive failures
        last_failure_time: Timestamp of last failure
        last_success_time: Timestamp of last success
    """

    def __init__(
        self,
        failure_threshold: int | None = None,
        timeout: int | None = None,
        half_open_timeout: int | None = None,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening (default from settings)
            timeout: Seconds before attempting recovery (default from settings)
            half_open_timeout: Seconds for half-open state (default from settings)
        """
        self.failure_threshold = (
            failure_threshold or settings.circuit_breaker_failure_threshold
        )
        self.timeout = timeout or settings.circuit_breaker_timeout
        self.half_open_timeout = (
            half_open_timeout or settings.circuit_breaker_half_open_timeout
        )

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.last_success_time: float | None = None
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Result from function execution

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception raised by the function
        """
        async with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info("Circuit breaker transitioning to HALF_OPEN state")
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                    )

        # Execute the function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset.

        Returns:
            True if circuit should transition to HALF_OPEN
        """
        if self.last_failure_time is None:
            return True

        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.timeout

    async def _on_success(self) -> None:
        """Handle successful function execution."""
        async with self._lock:
            self.failure_count = 0
            self.last_success_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker transitioning to CLOSED state after success")
                self.state = CircuitState.CLOSED

    async def _on_failure(self) -> None:
        """Handle failed function execution."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker transitioning to OPEN state after failure in HALF_OPEN")
                self.state = CircuitState.OPEN
            elif self.failure_count >= self.failure_threshold:
                logger.warning(
                    f"Circuit breaker transitioning to OPEN state after {self.failure_count} failures"
                )
                self.state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset to CLOSED state")

    def get_state(self) -> CircuitState:
        """Get current circuit state.

        Returns:
            Current circuit state
        """
        return self.state

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with circuit breaker stats
        """
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "timeout": self.timeout,
        }


def circuit_breaker(
    failure_threshold: int | None = None,
    timeout: int | None = None,
    half_open_timeout: int | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for applying circuit breaker to async functions.

    Args:
        failure_threshold: Number of failures before opening (default from settings)
        timeout: Seconds before attempting recovery (default from settings)
        half_open_timeout: Seconds for half-open state (default from settings)

    Returns:
        Decorated function with circuit breaker protection

    Example:
        @circuit_breaker(failure_threshold=5, timeout=60)
        async def call_external_service():
            ...
    """
    breaker = CircuitBreaker(
        failure_threshold=failure_threshold,
        timeout=timeout,
        half_open_timeout=half_open_timeout,
    )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await breaker.call(func, *args, **kwargs)

        # Attach breaker instance to wrapper for external access
        wrapper.circuit_breaker = breaker  # type: ignore[attr-defined]
        return wrapper

    return decorator
