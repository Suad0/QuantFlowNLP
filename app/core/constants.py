"""System constants and enumerations.

This module defines constants used throughout the application for consistency
and maintainability.
"""

from enum import Enum


class Sentiment(int, Enum):
    """Sentiment values for article analysis."""

    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1


class EventType(str, Enum):
    """Event types for news categorization."""

    EARNINGS = "earnings"
    MERGER = "merger"
    REGULATORY = "regulatory"
    PRODUCT_LAUNCH = "product_launch"
    GENERAL = "general"
    ECONOMIC_DATA = "economic_data"
    CENTRAL_BANK = "central_bank"
    GEOPOLITICAL = "geopolitical"


class NewsSource(str, Enum):
    """Supported news sources."""

    YAHOO_FINANCE = "yahoo"
    REUTERS = "reuters"
    BLOOMBERG = "bloomberg"
    INVESTING_COM = "investing"
    MARKETWATCH = "marketwatch"


class NormalizationMethod(str, Enum):
    """Feature normalization methods."""

    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"


class OptimizationMethod(str, Enum):
    """Portfolio optimization methods."""

    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    EFFICIENT_FRONTIER = "efficient_frontier"


class ConstraintType(str, Enum):
    """Portfolio constraint types."""

    WEIGHT_BOUNDS = "weight_bounds"
    LEVERAGE = "leverage"
    VARIANCE = "variance"
    SECTOR_EXPOSURE = "sector_exposure"


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# Database constants
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 1000

# Embedding dimensions
NOMIC_EMBED_DIMENSION = 768

# Score ranges
MIN_SCORE = -1.0
MAX_SCORE = 1.0

# Time constants
SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
TRADING_DAYS_PER_YEAR = 252

# Model constants
DEFAULT_BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 1000
