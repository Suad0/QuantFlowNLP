"""Unit tests for retry utilities."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.utils.retry import retry_async, retry_sync


@pytest.mark.asyncio
async def test_retry_async_success_first_attempt():
    """Test retry decorator with successful first attempt."""
    mock_func = AsyncMock(return_value="success")

    decorated = retry_async(max_attempts=3)(mock_func)
    result = await decorated()

    assert result == "success"
    assert mock_func.call_count == 1


@pytest.mark.asyncio
async def test_retry_async_success_after_failures():
    """Test retry decorator succeeds after initial failures."""
    call_count = 0

    async def sometimes_failing():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Temporary failure")
        return "success"

    decorated = retry_async(max_attempts=3, min_wait=0, max_wait=0)(sometimes_failing)
    result = await decorated()

    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_async_all_attempts_fail():
    """Test retry decorator when all attempts fail."""
    mock_func = AsyncMock(side_effect=ValueError("Persistent failure"))

    decorated = retry_async(max_attempts=3, min_wait=0, max_wait=0)(mock_func)

    with pytest.raises(ValueError, match="Persistent failure"):
        await decorated()

    assert mock_func.call_count == 3


@pytest.mark.asyncio
async def test_retry_async_specific_exceptions():
    """Test retry decorator only retries specific exceptions."""
    call_count = 0

    async def specific_exception_func():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Retryable error")
        raise TypeError("Non-retryable error")

    decorated = retry_async(
        max_attempts=3,
        min_wait=0,
        max_wait=0,
        exceptions=(ValueError,)
    )(specific_exception_func)

    with pytest.raises(TypeError, match="Non-retryable error"):
        await decorated()

    # Should have retried once for ValueError, then failed on TypeError
    assert call_count == 2


def test_retry_sync_success_first_attempt():
    """Test sync retry decorator with successful first attempt."""
    mock_func = MagicMock(return_value="success")

    decorated = retry_sync(max_attempts=3)(mock_func)
    result = decorated()

    assert result == "success"
    assert mock_func.call_count == 1


def test_retry_sync_success_after_failures():
    """Test sync retry decorator succeeds after initial failures."""
    call_count = 0

    def sometimes_failing():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Temporary failure")
        return "success"

    decorated = retry_sync(max_attempts=3, min_wait=0, max_wait=0)(sometimes_failing)
    result = decorated()

    assert result == "success"
    assert call_count == 3


def test_retry_sync_all_attempts_fail():
    """Test sync retry decorator when all attempts fail."""
    def failing_func():
        raise ValueError("Persistent failure")

    decorated = retry_sync(max_attempts=3, min_wait=0, max_wait=0)(failing_func)

    with pytest.raises(ValueError, match="Persistent failure"):
        decorated()


@pytest.mark.asyncio
async def test_retry_async_with_args_and_kwargs():
    """Test retry decorator preserves function arguments."""
    mock_func = AsyncMock(return_value="success")

    decorated = retry_async(max_attempts=3)(mock_func)
    result = await decorated("arg1", "arg2", kwarg1="value1")

    assert result == "success"
    mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")


def test_retry_sync_with_args_and_kwargs():
    """Test sync retry decorator preserves function arguments."""
    mock_func = MagicMock(return_value="success")

    decorated = retry_sync(max_attempts=3)(mock_func)
    result = decorated("arg1", "arg2", kwarg1="value1")

    assert result == "success"
    mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")
