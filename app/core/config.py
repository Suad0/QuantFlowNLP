"""Configuration management using Pydantic Settings.

This module provides centralized configuration management for the entire application,
loading settings from environment variables with validation and type safety.
"""

import json
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All configuration parameters are defined here with type hints, default values,
    and validation. Settings are loaded from environment variables or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database Configuration
    database_url: str = Field(
        default="sqlite:///data/trading_system.db",
        description="SQLite database URL",
    )

    # ChromaDB Configuration
    chromadb_host: str = Field(default="localhost", description="ChromaDB host")
    chromadb_port: int = Field(default=8001, description="ChromaDB port")
    chromadb_collection: str = Field(
        default="financial_articles",
        description="ChromaDB collection name",
    )
    chromadb_persist_directory: str = Field(
        default="./chroma_data",
        description="ChromaDB persistence directory",
    )
    chromadb_use_persistent: bool = Field(
        default=True,
        description="Use persistent local storage (True) or HTTP client (False)",
    )

    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    ollama_llm_model: str = Field(default="llama3", description="LLM model name")
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model name",
    )
    ollama_timeout: int = Field(default=60, description="Ollama request timeout in seconds")
    ollama_max_retries: int = Field(default=3, description="Maximum retry attempts for Ollama")

    # News Ingestion Configuration
    news_fetch_timeout: int = Field(default=30, description="RSS fetch timeout in seconds")
    news_article_fetch_timeout: int = Field(
        default=60,
        description="Article fetch timeout in seconds",
    )
    news_max_articles_per_source: int = Field(
        default=100,
        description="Maximum articles to fetch per source",
    )
    news_deduplication_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for deduplication",
    )
    news_embedding_batch_size: int = Field(
        default=10,
        description="Batch size for embedding generation",
    )

    # NLP Analysis Configuration
    nlp_summary_max_tokens: int = Field(
        default=200,
        description="Maximum tokens for article summaries",
    )
    nlp_analysis_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM analysis",
    )
    nlp_event_type_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "earnings": 1.5,
            "merger": 1.3,
            "regulatory": 1.2,
            "product_launch": 1.0,
            "general": 0.8,
        },
        description="Event type weights for news score computation",
    )

    # Feature Engineering Configuration
    feature_default_sequence_length: int = Field(
        default=168,
        description="Default sequence length (168 = 1 week hourly)",
    )
    feature_target_frequency: str = Field(
        default="1h",
        description="Target frequency for resampling",
    )
    feature_columns: list[str] = Field(
        default_factory=lambda: ["open", "high", "low", "close", "volume", "news_score"],
        description="Feature columns to include",
    )
    feature_normalization_method: str = Field(
        default="standard",
        description="Normalization method (standard, minmax, robust)",
    )

    # Prediction Configuration
    model_dir: str = Field(default="./models", description="Model storage directory")
    model_default_name: str = Field(
        default="xlstm_forecaster_v1.pkl",
        description="Default model filename",
    )
    prediction_inference_batch_size: int = Field(
        default=32,
        description="Batch size for inference",
    )
    prediction_timeout: int = Field(
        default=30,
        description="Prediction timeout in seconds",
    )

    # Portfolio Optimization Configuration
    portfolio_risk_free_rate: float = Field(
        default=0.02,
        description="Risk-free rate for portfolio optimization",
    )
    portfolio_default_method: str = Field(
        default="max_sharpe",
        description="Default optimization method",
    )
    portfolio_max_leverage: float = Field(
        default=1.0,
        description="Maximum portfolio leverage",
    )
    portfolio_min_weight: float = Field(
        default=0.0,
        description="Minimum asset weight",
    )
    portfolio_max_weight: float = Field(
        default=1.0,
        description="Maximum asset weight",
    )
    portfolio_rebalance_threshold: float = Field(
        default=0.05,
        description="Rebalance threshold",
    )

    # Quantitative Finance Configuration
    quant_default_risk_free_rate: float = Field(
        default=0.02,
        description="Default risk-free rate",
    )
    quant_default_volatility: float = Field(
        default=0.20,
        description="Default volatility",
    )
    quant_day_count_convention: str = Field(
        default="Actual/365",
        description="Day count convention",
    )
    quant_calendar: str = Field(
        default="UnitedStates", description="Calendar for date calculations"
    )

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    log_file: str = Field(
        default="logs/trading_system.log",
        description="Log file path",
    )
    log_max_bytes: int = Field(
        default=10485760,
        description="Maximum log file size in bytes",
    )
    log_backup_count: int = Field(default=5, description="Number of log backup files")

    # API Configuration
    api_title: str = Field(
        default="Quantitative Trading Intelligence System",
        description="API title",
    )
    api_version: str = Field(default="1.0.0", description="API version")
    api_description: str = Field(
        default="Local quantitative trading system with NLP analysis and portfolio optimization",
        description="API description",
    )
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"],
        description="CORS allowed origins",
    )
    api_cors_allow_credentials: bool = Field(
        default=True,
        description="CORS allow credentials",
    )
    api_cors_allow_methods: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed methods",
    )
    api_cors_allow_headers: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed headers",
    )

    # Circuit Breaker Configuration
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        description="Failure threshold for circuit breaker",
    )
    circuit_breaker_timeout: int = Field(
        default=60,
        description="Circuit breaker timeout in seconds",
    )
    circuit_breaker_half_open_timeout: int = Field(
        default=30,
        description="Half-open timeout in seconds",
    )

    # Retry Configuration
    retry_max_attempts: int = Field(default=3, description="Maximum retry attempts")
    retry_min_wait: int = Field(default=2, description="Minimum wait time between retries")
    retry_max_wait: int = Field(default=10, description="Maximum wait time between retries")
    retry_multiplier: int = Field(default=1, description="Exponential backoff multiplier")

    # Environment
    environment: str = Field(default="development", description="Environment name")

    @field_validator("nlp_event_type_weights", mode="before")
    @classmethod
    def parse_event_weights(cls, v: Any) -> dict[str, float]:
        """Parse event type weights from JSON string or dict."""
        if isinstance(v, str):
            return json.loads(v)  # type: ignore[no-any-return]
        return v  # type: ignore[no-any-return]

    @field_validator("feature_columns", mode="before")
    @classmethod
    def parse_feature_columns(cls, v: Any) -> list[str]:
        """Parse feature columns from JSON string or list."""
        if isinstance(v, str):
            return json.loads(v)  # type: ignore[no-any-return]
        return v  # type: ignore[no-any-return]

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        """Parse CORS origins from JSON string or list."""
        if isinstance(v, str):
            return json.loads(v)  # type: ignore[no-any-return]
        return v  # type: ignore[no-any-return]

    @field_validator("api_cors_allow_methods", mode="before")
    @classmethod
    def parse_cors_methods(cls, v: Any) -> list[str]:
        """Parse CORS methods from JSON string or list."""
        if isinstance(v, str):
            return json.loads(v)  # type: ignore[no-any-return]
        return v  # type: ignore[no-any-return]

    @field_validator("api_cors_allow_headers", mode="before")
    @classmethod
    def parse_cors_headers(cls, v: Any) -> list[str]:
        """Parse CORS headers from JSON string or list."""
        if isinstance(v, str):
            return json.loads(v)  # type: ignore[no-any-return]
        return v  # type: ignore[no-any-return]


# Global settings instance
settings = Settings()
