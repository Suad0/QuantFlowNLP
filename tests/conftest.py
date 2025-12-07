"""Pytest configuration and shared fixtures.

This module provides common fixtures and configuration for all tests,
including database setup, mock clients, and test data.
"""

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session.

    Yields:
        Event loop for async tests
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> dict:
    """Provide test settings.

    Returns:
        Dictionary of test configuration values
    """
    return {
        "database_url": "sqlite:///:memory:",
        "log_level": "DEBUG",
        "environment": "testing",
    }
