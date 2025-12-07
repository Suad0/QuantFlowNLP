"""Unit tests for circuit breaker."""

import asyncio
import pytest
from unittest.mock import AsyncMock

from app.utils.circuit_breaker import CircuitBreaker, CircuitState
from app.core.exceptions import CircuitBreakerOpenError


@pytest.fixture
def circuit_breaker():
    """Create circuit breaker for testing."""
    return CircuitBreaker(failure_threshold=3, timeout=1, half_open_timeout=1)


@pytest.mark.asyncio
async def test_circuit_breaker_initialization(circuit_breaker):
    """Test circuit breaker initialization."""
    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker.failure_count == 0
    assert circuit_breaker.failure_threshold == 3


@pytest.mark.asyncio
async def test_successful_call(circuit_breaker):
    """Test successful function call through circuit breaker."""
    async def success_func():
        return "success"

    result = await circuit_breaker.call(success_func)

    assert result == "success"
    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker.failure_count == 0


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold(circuit_breaker):
    """Test circuit opens after failure threshold is reached."""
    async def failing_func():
        raise Exception("Test failure")

    # Fail 3 times to reach threshold
    for _ in range(3):
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_func)

    # Circuit should now be open
    assert circuit_breaker.state == CircuitState.OPEN
    assert circuit_breaker.failure_count == 3


@pytest.mark.asyncio
async def test_circuit_blocks_when_open(circuit_breaker):
    """Test circuit breaker blocks calls when open."""
    async def failing_func():
        raise Exception("Test failure")

    # Open the circuit
    for _ in range(3):
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_func)

    # Next call should be blocked
    with pytest.raises(CircuitBreakerOpenError):
        await circuit_breaker.call(failing_func)


@pytest.mark.asyncio
async def test_circuit_transitions_to_half_open(circuit_breaker):
    """Test circuit transitions to half-open after timeout."""
    async def failing_func():
        raise Exception("Test failure")

    # Open the circuit
    for _ in range(3):
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_func)

    assert circuit_breaker.state == CircuitState.OPEN

    # Wait for timeout
    await asyncio.sleep(1.1)

    # Next call should transition to half-open
    with pytest.raises(Exception):
        await circuit_breaker.call(failing_func)

    # Should be back to open after failure in half-open
    assert circuit_breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_closes_after_success_in_half_open(circuit_breaker):
    """Test circuit closes after successful call in half-open state."""
    call_count = 0

    async def sometimes_failing_func():
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            raise Exception("Test failure")
        return "success"

    # Open the circuit
    for _ in range(3):
        with pytest.raises(Exception):
            await circuit_breaker.call(sometimes_failing_func)

    assert circuit_breaker.state == CircuitState.OPEN

    # Wait for timeout
    await asyncio.sleep(1.1)

    # Successful call should close the circuit
    result = await circuit_breaker.call(sometimes_failing_func)

    assert result == "success"
    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker.failure_count == 0


@pytest.mark.asyncio
async def test_manual_reset(circuit_breaker):
    """Test manual circuit breaker reset."""
    async def failing_func():
        raise Exception("Test failure")

    # Open the circuit
    for _ in range(3):
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_func)

    assert circuit_breaker.state == CircuitState.OPEN

    # Manual reset
    circuit_breaker.reset()

    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker.failure_count == 0


@pytest.mark.asyncio
async def test_get_stats(circuit_breaker):
    """Test getting circuit breaker statistics."""
    stats = circuit_breaker.get_stats()

    assert "state" in stats
    assert "failure_count" in stats
    assert "failure_threshold" in stats
    assert "timeout" in stats
    assert stats["state"] == "closed"
    assert stats["failure_count"] == 0


@pytest.mark.asyncio
async def test_success_resets_failure_count(circuit_breaker):
    """Test that successful calls reset failure count."""
    call_count = 0

    async def alternating_func():
        nonlocal call_count
        call_count += 1
        if call_count in [1, 2]:
            raise Exception("Test failure")
        return "success"

    # Two failures
    for _ in range(2):
        with pytest.raises(Exception):
            await circuit_breaker.call(alternating_func)

    assert circuit_breaker.failure_count == 2

    # Success should reset count
    result = await circuit_breaker.call(alternating_func)

    assert result == "success"
    assert circuit_breaker.failure_count == 0
    assert circuit_breaker.state == CircuitState.CLOSED
