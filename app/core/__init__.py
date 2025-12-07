"""Core functionality for the trading system."""

from app.core.config import settings
from app.core.exceptions import (
    AnalysisRepositoryError,
    ArticleRepositoryError,
    CircuitBreakerOpenError,
    DatabaseError,
    FeatureEngineeringError,
    NewsIngestionError,
    NLPAnalysisError,
    OHLCVRepositoryError,
    OllamaConnectionError,
    PortfolioOptimizationError,
    PredictionError,
    QuantitativeFinanceError,
    ScalerRepositoryError,
    TradingSystemError,
    VectorRepositoryError,
)

__all__ = [
    "settings",
    "TradingSystemError",
    "NewsIngestionError",
    "NLPAnalysisError",
    "FeatureEngineeringError",
    "PredictionError",
    "PortfolioOptimizationError",
    "QuantitativeFinanceError",
    "DatabaseError",
    "OllamaConnectionError",
    "ArticleRepositoryError",
    "AnalysisRepositoryError",
    "VectorRepositoryError",
    "OHLCVRepositoryError",
    "ScalerRepositoryError",
    "CircuitBreakerOpenError",
]
