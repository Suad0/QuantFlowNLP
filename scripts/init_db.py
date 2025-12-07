#!/usr/bin/env python3
"""Database initialization script.

This script initializes the SQLite database with the required schema,
creates all tables and indexes, and optionally seeds test data.

Usage:
    python scripts/init_db.py [--seed]
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.adapters.database import DatabaseManager
from app.core.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


async def init_database(seed: bool = False) -> None:
    """Initialize database with schema.
    
    Args:
        seed: Whether to seed test data after initialization
    """
    logger.info("Starting database initialization...")
    logger.info(f"Database URL: {settings.database_url}")
    
    db_manager = DatabaseManager()
    
    try:
        # Connect to database
        await db_manager.connect()
        logger.info("Database connection established")
        
        # Initialize schema
        await db_manager.initialize_schema()
        logger.info("Database schema initialized successfully")
        
        # Verify tables were created
        conn = await db_manager.get_connection()
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = await cursor.fetchall()
        await cursor.close()
        
        logger.info("Created tables:")
        for table in tables:
            logger.info(f"  - {table[0]}")
        
        # Seed test data if requested
        if seed:
            logger.info("Seeding test data...")
            await seed_test_data(conn)
            logger.info("Test data seeded successfully")
        
        logger.info("Database initialization complete!")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    finally:
        await db_manager.disconnect()


async def seed_test_data(conn) -> None:  # type: ignore[no-untyped-def]
    """Seed database with test data.
    
    Args:
        conn: Database connection
    """
    from datetime import datetime, timezone
    import uuid
    
    # Insert sample article
    article_id = str(uuid.uuid4())
    await conn.execute(
        """
        INSERT OR IGNORE INTO articles 
        (id, title, content, source, url, published_at, fetched_at, is_duplicate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            article_id,
            "Sample Financial News Article",
            "This is a sample article content for testing purposes. "
            "It contains information about market movements and economic indicators.",
            "reuters",
            f"https://example.com/article/{article_id}",
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(),
            False,
        ),
    )
    
    # Insert sample OHLCV data
    ohlcv_id = str(uuid.uuid4())
    await conn.execute(
        """
        INSERT OR IGNORE INTO ohlcv_data
        (id, symbol, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ohlcv_id,
            "AAPL",
            datetime.now(timezone.utc).isoformat(),
            150.0,
            152.5,
            149.5,
            151.0,
            1000000.0,
        ),
    )
    
    # Insert sample scaler params
    scaler_id = str(uuid.uuid4())
    await conn.execute(
        """
        INSERT OR IGNORE INTO scaler_params
        (id, symbol, feature_name, mean, std, min, max, method, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            scaler_id,
            "AAPL",
            "close",
            150.0,
            10.0,
            100.0,
            200.0,
            "standard",
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    
    await conn.commit()
    logger.info(f"Inserted sample article: {article_id}")
    logger.info(f"Inserted sample OHLCV data: {ohlcv_id}")
    logger.info(f"Inserted sample scaler params: {scaler_id}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Initialize trading system database")
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed database with test data",
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(init_database(seed=args.seed))
        sys.exit(0)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
