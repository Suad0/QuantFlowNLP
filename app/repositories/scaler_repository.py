"""SQLite implementation of ScalerRepository.

This module provides the concrete implementation of scaler parameters storage
using SQLite with aiosqlite for async operations.
"""

import aiosqlite
from datetime import datetime
from typing import List, Optional

from app.core.exceptions import ScalerRepositoryError
from app.models.domain import ScalerParams
from app.repositories.base import ScalerRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteScalerRepository(ScalerRepository):
    """SQLite implementation of ScalerRepository.
    
    Provides async operations for storing and retrieving normalization parameters
    from SQLite database.
    """

    def __init__(self, connection: aiosqlite.Connection):
        """Initialize repository with database connection.
        
        Args:
            connection: Active aiosqlite connection
        """
        self.connection = connection

    async def save_params(self, params: ScalerParams) -> str:
        """Save scaler parameters.
        
        Args:
            params: Scaler parameters to save
            
        Returns:
            Parameters ID
            
        Raises:
            ScalerRepositoryError: If save fails
        """
        try:
            # Use INSERT OR REPLACE to handle updates
            await self.connection.execute(
                """
                INSERT OR REPLACE INTO scaler_params 
                (id, symbol, feature_name, mean, std, min, max, method, 
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    params.id,
                    params.symbol,
                    params.feature_name,
                    params.mean,
                    params.std,
                    params.min,
                    params.max,
                    params.method,
                    params.created_at.isoformat(),
                    params.updated_at.isoformat(),
                ),
            )
            await self.connection.commit()
            logger.debug(
                f"Saved scaler params: {params.id} for {params.symbol}/{params.feature_name}"
            )
            return params.id
            
        except Exception as e:
            logger.error(f"Failed to save scaler params: {e}")
            raise ScalerRepositoryError(f"Failed to save scaler params: {e}") from e

    async def get_params(
        self,
        symbol: str,
        feature_name: str,
    ) -> Optional[ScalerParams]:
        """Get scaler parameters for a symbol and feature.
        
        Args:
            symbol: Trading symbol
            feature_name: Feature name
            
        Returns:
            Scaler parameters if found, None otherwise
        """
        try:
            cursor = await self.connection.execute(
                """
                SELECT id, symbol, feature_name, mean, std, min, max, method,
                       created_at, updated_at
                FROM scaler_params
                WHERE symbol = ? AND feature_name = ?
                """,
                (symbol, feature_name),
            )
            row = await cursor.fetchone()
            await cursor.close()
            
            if row:
                return self._row_to_params(row)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get scaler params for {symbol}/{feature_name}: {e}")
            raise ScalerRepositoryError(f"Failed to get scaler params: {e}") from e

    async def get_all_params(self, symbol: str) -> List[ScalerParams]:
        """Get all scaler parameters for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of scaler parameters
        """
        try:
            cursor = await self.connection.execute(
                """
                SELECT id, symbol, feature_name, mean, std, min, max, method,
                       created_at, updated_at
                FROM scaler_params
                WHERE symbol = ?
                ORDER BY feature_name
                """,
                (symbol,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            
            return [self._row_to_params(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get all scaler params for {symbol}: {e}")
            raise ScalerRepositoryError(f"Failed to get all scaler params: {e}") from e

    async def update_params(self, params: ScalerParams) -> None:
        """Update existing scaler parameters.
        
        Args:
            params: Updated scaler parameters
            
        Raises:
            ScalerRepositoryError: If update fails
        """
        try:
            # Update the updated_at timestamp
            params.updated_at = datetime.utcnow()
            
            await self.connection.execute(
                """
                UPDATE scaler_params
                SET mean = ?, std = ?, min = ?, max = ?, method = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    params.mean,
                    params.std,
                    params.min,
                    params.max,
                    params.method,
                    params.updated_at.isoformat(),
                    params.id,
                ),
            )
            await self.connection.commit()
            logger.debug(f"Updated scaler params: {params.id}")
            
        except Exception as e:
            logger.error(f"Failed to update scaler params {params.id}: {e}")
            raise ScalerRepositoryError(f"Failed to update scaler params: {e}") from e

    async def delete_params(self, symbol: str, feature_name: str) -> None:
        """Delete scaler parameters.
        
        Args:
            symbol: Trading symbol
            feature_name: Feature name
            
        Raises:
            ScalerRepositoryError: If deletion fails
        """
        try:
            await self.connection.execute(
                """
                DELETE FROM scaler_params
                WHERE symbol = ? AND feature_name = ?
                """,
                (symbol, feature_name),
            )
            await self.connection.commit()
            logger.debug(f"Deleted scaler params for {symbol}/{feature_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete scaler params for {symbol}/{feature_name}: {e}")
            raise ScalerRepositoryError(f"Failed to delete scaler params: {e}") from e

    def _row_to_params(self, row: tuple) -> ScalerParams:  # type: ignore[type-arg]
        """Convert database row to ScalerParams object.
        
        Args:
            row: Database row tuple
            
        Returns:
            ScalerParams object
        """
        return ScalerParams(
            id=row[0],
            symbol=row[1],
            feature_name=row[2],
            mean=row[3],
            std=row[4],
            min=row[5],
            max=row[6],
            method=row[7],
            created_at=datetime.fromisoformat(row[8]),
            updated_at=datetime.fromisoformat(row[9]),
        )
