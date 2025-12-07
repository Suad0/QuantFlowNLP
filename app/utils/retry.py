"""Retry utilities with exponential backoff.

This module provides decorators and utilities for retrying operations
with configurable backoff strategies.
"""

import asyncio
import functools
import logging
from typing import Any, Callable, Type, TypeVar

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_async(
    max_attempts: int | None = None,
    min_wait: int | None = None,
    max_wait: int | None = None,
    multiplier: int | None = None,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for retrying async functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts (default from settings)
        min_wait: Minimum wait time in seconds (default from settings)
        max_wait: Maximum wait time in seconds (default from settings)
        multiplier: Exponential backoff multiplier (default from settings)
        exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function with retry logic

    Example:
        @retry_async(max_attempts=3, exceptions=(httpx.RequestError,))
        async def fetch_data():
            ...
    """
    # Use settings defaults if not provided
    _max_attempts = max_attempts or settings.retry_max_attempts
    _min_wait = min_wait or settings.retry_min_wait
    _max_wait = max_wait or settings.retry_max_wait
    _multiplier = multiplier or settings.retry_multiplier

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(_max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < _max_attempts - 1:
                        # Calculate wait time with exponential backoff
                        wait_time = min(
                            _min_wait * (_multiplier ** attempt),
                            _max_wait,
                        )

                        logger.warning(
                            f"Attempt {attempt + 1}/{_max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {wait_time}s..."
                        )

                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"All {_max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            # Raise the last exception if all retries failed
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator


def retry_sync(
    max_attempts: int | None = None,
    min_wait: int | None = None,
    max_wait: int | None = None,
    multiplier: int | None = None,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for retrying synchronous functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts (default from settings)
        min_wait: Minimum wait time in seconds (default from settings)
        max_wait: Maximum wait time in seconds (default from settings)
        multiplier: Exponential backoff multiplier (default from settings)
        exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function with retry logic

    Example:
        @retry_sync(max_attempts=3, exceptions=(ConnectionError,))
        def fetch_data():
            ...
    """
    # Use settings defaults if not provided
    _max_attempts = max_attempts or settings.retry_max_attempts
    _min_wait = min_wait or settings.retry_min_wait
    _max_wait = max_wait or settings.retry_max_wait
    _multiplier = multiplier or settings.retry_multiplier

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import time

            last_exception = None

            for attempt in range(_max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < _max_attempts - 1:
                        # Calculate wait time with exponential backoff
                        wait_time = min(
                            _min_wait * (_multiplier ** attempt),
                            _max_wait,
                        )

                        logger.warning(
                            f"Attempt {attempt + 1}/{_max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {wait_time}s..."
                        )

                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"All {_max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            # Raise the last exception if all retries failed
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator
