"""SQLite implementation of OHLCVRepository.

This module provides the concrete implementation of OHLCV market data storage
using SQLite with aiosqlite for async operations.
"""

import aiosqlite
from datetime import datetime
from typing import List

from app.core.exceptions import OHLCVRepositoryError
from app.models.domain import OHLCVData
from app.repositories.base import OHLCVRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteOHLCVRepository(OHLCVRepository):
    """SQLite implementation of OHLCVRepository.
    
    Provides async operations for storing and retrieving OHLCV market data
    from SQLite database.
    """

    def __init__(self, connection: aiosqlite.Connection):
        """Initialize repository with database connection.
        
        Args:
            connection: Active aiosqlite connection
        """
        self.connection = connection

    async def insert(self, data: OHLCVData) -> str:
        """Insert OHLCV data point.
        
        Args:
            data: OHLCV data to insert
            
        Returns:
            Data ID
            
        Raises:
            OHLCVRepositoryError: If insertion fails
        """
        try:
            await self.connection.execute(
                """
                INSERT INTO ohlcv_data 
                (id, symbol, timestamp, open, high, low, close, volume, 
                 adjusted_close, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.id,
                    data.symbol,
                    data.timestamp.isoformat(),
                    data.open,
                    data.high,
                    data.low,
                    data.close,
                    data.volume,
                    data.adjusted_close,
                    data.metadata,
                ),
            )
            await self.connection.commit()
            logger.debug(f"Inserted OHLCV data: {data.id} for {data.symbol}")
            return data.id
            
        except aiosqlite.IntegrityError as e:
            logger.error(f"OHLCV data already exists or constraint violation: {e}")
            raise OHLCVRepositoryError(f"Failed to insert OHLCV data: {e}") from e
        except Exception as e:
            logger.error(f"Failed to insert OHLCV data: {e}")
            raise OHLCVRepositoryError(f"Failed to insert OHLCV data: {e}") from e

    async def insert_batch(self, data_list: List[OHLCVData]) -> int:
        """Insert multiple OHLCV data points.
        
        Args:
            data_list: List of OHLCV data
            
        Returns:
            Number of records inserted
            
        Raises:
            OHLCVRepositoryError: If batch insertion fails
        """
        if not data_list:
            return 0
        
        try:
            values = [
                (
                    data.id,
                    data.symbol,
                    data.timestamp.isoformat(),
                    data.open,
                    data.high,
                    data.low,
                    data.close,
                    data.volume,
                    data.adjusted_close,
                    data.metadata,
                )
                for data in data_list
            ]
            
            await self.connection.executemany(
                """
                INSERT OR IGNORE INTO ohlcv_data 
                (id, symbol, timestamp, open, high, low, close, volume, 
                 adjusted_close, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            await self.connection.commit()
            
            logger.debug(f"Inserted {len(data_list)} OHLCV data points")
            return len(data_list)
            
        except Exception as e:
            logger.error(f"Failed to insert OHLCV batch: {e}")
            raise OHLCVRepositoryError(f"Failed to insert OHLCV batch: {e}") from e

    async def get_range(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[OHLCVData]:
        """Get OHLCV data for a time range.
        
        Args:
            symbol: Trading symbol
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of OHLCV data points
        """
        try:
            cursor = await self.connection.execute(
                """
                SELECT id, symbol, timestamp, open, high, low, close, volume,
                       adjusted_close, metadata
                FROM ohlcv_data
                WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                """,
                (symbol, start_time.isoformat(), end_time.isoformat()),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            
            return [self._row_to_ohlcv(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get OHLCV range for {symbol}: {e}")
            raise OHLCVRepositoryError(f"Failed to get OHLCV range: {e}") from e

    async def get_latest(self, symbol: str, limit: int = 1) -> List[OHLCVData]:
        """Get latest OHLCV data points.
        
        Args:
            symbol: Trading symbol
            limit: Number of latest points to retrieve
            
        Returns:
            List of latest OHLCV data points
        """
        try:
            cursor = await self.connection.execute(
                """
                SELECT id, symbol, timestamp, open, high, low, close, volume,
                       adjusted_close, metadata
                FROM ohlcv_data
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (symbol, limit),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            
            return [self._row_to_ohlcv(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get latest OHLCV for {symbol}: {e}")
            raise OHLCVRepositoryError(f"Failed to get latest OHLCV: {e}") from e

    async def delete_range(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Delete OHLCV data for a time range.
        
        Args:
            symbol: Trading symbol
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            Number of records deleted
        """
        try:
            cursor = await self.connection.execute(
                """
                DELETE FROM ohlcv_data
                WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
                """,
                (symbol, start_time.isoformat(), end_time.isoformat()),
            )
            await self.connection.commit()
            
            deleted_count = cursor.rowcount
            await cursor.close()
            
            logger.debug(f"Deleted {deleted_count} OHLCV records for {symbol}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete OHLCV range for {symbol}: {e}")
            raise OHLCVRepositoryError(f"Failed to delete OHLCV range: {e}") from e

    def _row_to_ohlcv(self, row: tuple) -> OHLCVData:  # type: ignore[type-arg]
        """Convert database row to OHLCVData object.
        
        Args:
            row: Database row tuple
            
        Returns:
            OHLCVData object
        """
        return OHLCVData(
            id=row[0],
            symbol=row[1],
            timestamp=datetime.fromisoformat(row[2]),
            open=row[3],
            high=row[4],
            low=row[5],
            close=row[6],
            volume=row[7],
            adjusted_close=row[8],
            metadata=row[9],
        )
