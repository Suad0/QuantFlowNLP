"""Custom exception hierarchy for the trading system.

This module defines all custom exceptions used throughout the application,
organized in a clear hierarchy for better error handling and reporting.
"""


class TradingSystemError(Exception):
    """Base exception for all trading system errors.

    All custom exceptions in the system should inherit from this base class
    to allow for consistent error handling at the API boundary.
    """

    pass


class NewsIngestionError(TradingSystemError):
    """Errors during news fetching and ingestion.

    Raised when there are issues fetching articles from news sources,
    parsing RSS feeds, or storing articles in the database.
    """

    pass


class NLPAnalysisError(TradingSystemError):
    """Errors during NLP processing.

    Raised when there are issues with LLM inference, embedding generation,
    or structured data extraction from articles.
    """

    pass


class FeatureEngineeringError(TradingSystemError):
    """Errors during feature preparation.

    Raised when there are issues building feature sequences, normalizing data,
    or aligning time series data.
    """

    pass


class PredictionError(TradingSystemError):
    """Errors during model inference.

    Raised when there are issues loading models, running predictions,
    or validating prediction inputs.
    """

    pass


class PortfolioOptimizationError(TradingSystemError):
    """Errors during portfolio optimization.

    Raised when optimization problems are infeasible, constraints are violated,
    or there are issues with optimization algorithms.
    """

    pass


class QuantitativeFinanceError(TradingSystemError):
    """Errors during quantitative finance calculations.

    Raised when there are issues with option pricing, bond valuation,
    or yield curve construction.
    """

    pass


class DatabaseError(TradingSystemError):
    """Database operation errors.

    Raised when there are issues with database connections, queries,
    or transactions.
    """

    pass


class OllamaConnectionError(TradingSystemError):
    """Ollama service connection errors.

    Raised when the Ollama service is unavailable, times out,
    or returns unexpected responses.
    """

    pass


class ArticleRepositoryError(DatabaseError):
    """Errors specific to article repository operations.

    Raised when there are issues creating, retrieving, or updating articles
    in the database.
    """

    pass


class AnalysisRepositoryError(DatabaseError):
    """Errors specific to analysis repository operations.

    Raised when there are issues storing or retrieving article analysis data.
    """

    pass


class VectorRepositoryError(DatabaseError):
    """Errors specific to vector repository operations.

    Raised when there are issues with ChromaDB operations, embedding storage,
    or similarity searches.
    """

    pass


class OHLCVRepositoryError(DatabaseError):
    """Errors specific to OHLCV repository operations.

    Raised when there are issues retrieving or storing OHLCV market data.
    """

    pass


class ScalerRepositoryError(DatabaseError):
    """Errors specific to scaler repository operations.

    Raised when there are issues storing or retrieving normalization parameters.
    """

    pass


class CircuitBreakerOpenError(TradingSystemError):
    """Circuit breaker is open, preventing requests.

    Raised when the circuit breaker is in the open state and blocking
    requests to prevent cascade failures.
    """

    pass
