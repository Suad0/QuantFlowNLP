"""Database connection management and initialization.

This module provides async database connection management using aiosqlite
and includes the SQL schema for all tables with proper indexes.
"""

import aiosqlite
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.core.exceptions import DatabaseError
from app.utils.logging import get_logger

logger = get_logger(__name__)


# SQL Schema Definitions
CREATE_ARTICLES_TABLE = """
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    source TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    published_at TIMESTAMP NOT NULL,
    fetched_at TIMESTAMP NOT NULL,
    is_duplicate BOOLEAN DEFAULT FALSE,
    duplicate_of TEXT,
    metadata TEXT,
    FOREIGN KEY (duplicate_of) REFERENCES articles(id)
);
"""

CREATE_ARTICLES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);",
    "CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);",
    "CREATE INDEX IF NOT EXISTS idx_articles_duplicate ON articles(is_duplicate);",
    "CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);",
]

CREATE_ARTICLE_ANALYSIS_TABLE = """
CREATE TABLE IF NOT EXISTS article_analysis (
    id TEXT PRIMARY KEY,
    article_id TEXT NOT NULL,
    summary TEXT,
    sentiment INTEGER NOT NULL,
    impact_magnitude REAL NOT NULL,
    event_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    estimated_price_move REAL NOT NULL,
    news_score REAL NOT NULL,
    analyzed_at TIMESTAMP NOT NULL,
    metadata TEXT,
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);
"""

CREATE_ARTICLE_ANALYSIS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_analysis_article ON article_analysis(article_id);",
    "CREATE INDEX IF NOT EXISTS idx_analysis_score ON article_analysis(news_score);",
    "CREATE INDEX IF NOT EXISTS idx_analysis_analyzed_at ON article_analysis(analyzed_at);",
]

CREATE_OHLCV_DATA_TABLE = """
CREATE TABLE IF NOT EXISTS ohlcv_data (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    adjusted_close REAL,
    metadata TEXT,
    UNIQUE(symbol, timestamp)
);
"""

CREATE_OHLCV_DATA_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv_data(symbol, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv_data(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv_data(timestamp);",
]

CREATE_SCALER_PARAMS_TABLE = """
CREATE TABLE IF NOT EXISTS scaler_params (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    mean REAL NOT NULL,
    std REAL NOT NULL,
    min REAL,
    max REAL,
    method TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    UNIQUE(symbol, feature_name)
);
"""

CREATE_SCALER_PARAMS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_scaler_symbol ON scaler_params(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_scaler_symbol_feature ON scaler_params(symbol, feature_name);",
]


class DatabaseManager:
    """Manages database connections and initialization.
    
    This class provides async context manager support for database connections
    and handles schema initialization with proper error handling.
    """

    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            database_url: SQLite database URL. If None, uses settings.database_url
        """
        self.database_url = database_url or settings.database_url
        # Extract file path from sqlite:/// URL
        if self.database_url.startswith("sqlite:///"):
            self.db_path = self.database_url.replace("sqlite:///", "")
        else:
            self.db_path = self.database_url
        
        self._connection: Optional[aiosqlite.Connection] = None

    async def connect(self) -> aiosqlite.Connection:
        """Establish database connection.
        
        Returns:
            Active database connection
            
        Raises:
            DatabaseError: If connection fails
        """
        try:
            # Ensure directory exists
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect with optimized settings
            self._connection = await aiosqlite.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False,
            )
            
            # Enable foreign keys
            await self._connection.execute("PRAGMA foreign_keys = ON;")
            
            # Set journal mode to WAL for better concurrency
            await self._connection.execute("PRAGMA journal_mode = WAL;")
            
            # Set synchronous mode to NORMAL for better performance
            await self._connection.execute("PRAGMA synchronous = NORMAL;")
            
            await self._connection.commit()
            
            logger.info(f"Connected to database: {self.db_path}")
            return self._connection
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise DatabaseError(f"Database connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._connection:
            try:
                await self._connection.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
            finally:
                self._connection = None

    async def initialize_schema(self) -> None:
        """Initialize database schema with all tables and indexes.
        
        Creates all required tables and indexes if they don't exist.
        This is idempotent and safe to call multiple times.
        
        Raises:
            DatabaseError: If schema initialization fails
        """
        if not self._connection:
            await self.connect()
        
        try:
            logger.info("Initializing database schema...")
            
            # Create tables
            await self._connection.execute(CREATE_ARTICLES_TABLE)
            logger.debug("Created articles table")
            
            await self._connection.execute(CREATE_ARTICLE_ANALYSIS_TABLE)
            logger.debug("Created article_analysis table")
            
            await self._connection.execute(CREATE_OHLCV_DATA_TABLE)
            logger.debug("Created ohlcv_data table")
            
            await self._connection.execute(CREATE_SCALER_PARAMS_TABLE)
            logger.debug("Created scaler_params table")
            
            # Create indexes
            for index_sql in CREATE_ARTICLES_INDEXES:
                await self._connection.execute(index_sql)
            logger.debug("Created articles indexes")
            
            for index_sql in CREATE_ARTICLE_ANALYSIS_INDEXES:
                await self._connection.execute(index_sql)
            logger.debug("Created article_analysis indexes")
            
            for index_sql in CREATE_OHLCV_DATA_INDEXES:
                await self._connection.execute(index_sql)
            logger.debug("Created ohlcv_data indexes")
            
            for index_sql in CREATE_SCALER_PARAMS_INDEXES:
                await self._connection.execute(index_sql)
            logger.debug("Created scaler_params indexes")
            
            await self._connection.commit()
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise DatabaseError(f"Schema initialization failed: {e}") from e

    async def get_connection(self) -> aiosqlite.Connection:
        """Get active database connection.
        
        Returns:
            Active database connection
            
        Raises:
            DatabaseError: If no active connection
        """
        if not self._connection:
            await self.connect()
        
        if not self._connection:
            raise DatabaseError("No active database connection")
        
        return self._connection

    async def __aenter__(self) -> aiosqlite.Connection:
        """Async context manager entry."""
        return await self.connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        """Async context manager exit."""
        await self.disconnect()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


async def get_database() -> aiosqlite.Connection:
    """Get database connection for dependency injection.
    
    This function is used as a FastAPI dependency to provide
    database connections to route handlers.
    
    Returns:
        Active database connection
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.connect()
        await _db_manager.initialize_schema()
    
    return await _db_manager.get_connection()


async def close_database() -> None:
    """Close global database connection.
    
    This should be called during application shutdown.
    """
    global _db_manager
    
    if _db_manager:
        await _db_manager.disconnect()
        _db_manager = None
