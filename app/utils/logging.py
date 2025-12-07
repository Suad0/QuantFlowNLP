"""Structured logging configuration with JSON formatting.

This module provides centralized logging setup with support for both JSON
and text formats, rotating file handlers, and structured log messages.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from pythonjsonlogger import jsonlogger  # type: ignore[import-untyped, unused-ignore]

from app.core.config import settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):  # type: ignore[name-defined, misc]
    """Custom JSON formatter with additional context fields."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        """Add custom fields to log records.

        Args:
            log_record: The log record dictionary to modify
            record: The original LogRecord object
            message_dict: Additional message fields
        """
        super().add_fields(log_record, record, message_dict)

        # Add standard fields
        log_record["timestamp"] = self.formatTime(record, self.datefmt)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["module"] = record.module
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno

        # Add environment
        log_record["environment"] = settings.environment

        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)


def setup_logging() -> None:
    """Configure application-wide logging.

    Sets up logging with both console and file handlers, using JSON or text
    format based on configuration. Creates log directory if it doesn't exist.
    """
    # Create logs directory if it doesn't exist
    log_file_path = Path(settings.log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level.upper()))

    # File handler with rotation
    file_handler = RotatingFileHandler(
        settings.log_file,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
    )
    file_handler.setLevel(getattr(logging, settings.log_level.upper()))

    # Set formatters based on configuration
    if settings.log_format == "json":
        # JSON formatter for structured logging
        json_formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(json_formatter)
        file_handler.setFormatter(json_formatter)
    else:
        # Text formatter for human-readable logs
        text_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(text_formatter)
        file_handler.setFormatter(text_formatter)

    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str, extra: dict[str, Any] | None = None) -> logging.LoggerAdapter[logging.Logger]:
    """Get a logger with optional extra context.

    Args:
        name: Logger name (typically __name__)
        extra: Additional context to include in all log messages

    Returns:
        LoggerAdapter with extra context
    """
    logger = logging.getLogger(name)

    if extra:
        return logging.LoggerAdapter(logger, extra)

    return logging.LoggerAdapter(logger, {})


class LogContext:
    """Context manager for adding temporary context to log messages.

    Example:
        with LogContext(request_id="abc123"):
            logger.info("Processing request")
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize log context.

        Args:
            **kwargs: Context fields to add to log messages
        """
        self.context = kwargs
        self.old_factory = logging.getLogRecordFactory()

    def __enter__(self) -> "LogContext":
        """Enter context and modify log record factory."""

        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and restore original log record factory."""
        logging.setLogRecordFactory(self.old_factory)
