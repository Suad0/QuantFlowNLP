"""SQLite implementation of AnalysisRepository.

This module provides the concrete implementation of article analysis storage
using SQLite with aiosqlite for async operations.
"""

import aiosqlite
from datetime import datetime
from typing import List, Optional

from app.core.exceptions import AnalysisRepositoryError
from app.models.domain import ArticleAnalysis
from app.repositories.base import AnalysisRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteAnalysisRepository(AnalysisRepository):
    """SQLite implementation of AnalysisRepository.
    
    Provides async operations for storing and retrieving article analysis
    from SQLite database.
    """

    def __init__(self, connection: aiosqlite.Connection):
        """Initialize repository with database connection.
        
        Args:
            connection: Active aiosqlite connection
        """
        self.connection = connection

    async def create(self, analysis: ArticleAnalysis) -> str:
        """Store article analysis.
        
        Args:
            analysis: Analysis to store
            
        Returns:
            Analysis ID
            
        Raises:
            AnalysisRepositoryError: If creation fails
        """
        try:
            await self.connection.execute(
                """
                INSERT INTO article_analysis 
                (id, article_id, summary, sentiment, impact_magnitude, event_type,
                 confidence, estimated_price_move, news_score, analyzed_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis.id,
                    analysis.article_id,
                    analysis.summary,
                    analysis.sentiment,
                    analysis.impact_magnitude,
                    analysis.event_type,
                    analysis.confidence,
                    analysis.estimated_price_move,
                    analysis.news_score,
                    analysis.analyzed_at.isoformat(),
                    analysis.metadata,
                ),
            )
            await self.connection.commit()
            logger.debug(f"Created analysis: {analysis.id} for article: {analysis.article_id}")
            return analysis.id
            
        except aiosqlite.IntegrityError as e:
            logger.error(f"Analysis already exists or constraint violation: {e}")
            raise AnalysisRepositoryError(f"Failed to create analysis: {e}") from e
        except Exception as e:
            logger.error(f"Failed to create analysis: {e}")
            raise AnalysisRepositoryError(f"Failed to create analysis: {e}") from e

    async def get_by_id(self, analysis_id: str) -> Optional[ArticleAnalysis]:
        """Retrieve analysis by ID.
        
        Args:
            analysis_id: Analysis identifier
            
        Returns:
            Analysis if found, None otherwise
        """
        try:
            cursor = await self.connection.execute(
                """
                SELECT id, article_id, summary, sentiment, impact_magnitude, event_type,
                       confidence, estimated_price_move, news_score, analyzed_at, metadata
                FROM article_analysis
                WHERE id = ?
                """,
                (analysis_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            
            if row:
                return self._row_to_analysis(row)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get analysis {analysis_id}: {e}")
            raise AnalysisRepositoryError(f"Failed to get analysis: {e}") from e

    async def get_by_article_id(self, article_id: str) -> Optional[ArticleAnalysis]:
        """Retrieve analysis for a specific article.
        
        Args:
            article_id: Article identifier
            
        Returns:
            Analysis if found, None otherwise
        """
        try:
            cursor = await self.connection.execute(
                """
                SELECT id, article_id, summary, sentiment, impact_magnitude, event_type,
                       confidence, estimated_price_move, news_score, analyzed_at, metadata
                FROM article_analysis
                WHERE article_id = ?
                """,
                (article_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            
            if row:
                return self._row_to_analysis(row)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get analysis for article {article_id}: {e}")
            raise AnalysisRepositoryError(f"Failed to get analysis by article: {e}") from e

    async def update(self, analysis: ArticleAnalysis) -> None:
        """Update existing analysis.
        
        Args:
            analysis: Analysis with updated values
            
        Raises:
            AnalysisRepositoryError: If update fails
        """
        try:
            await self.connection.execute(
                """
                UPDATE article_analysis
                SET summary = ?, sentiment = ?, impact_magnitude = ?, event_type = ?,
                    confidence = ?, estimated_price_move = ?, news_score = ?,
                    analyzed_at = ?, metadata = ?
                WHERE id = ?
                """,
                (
                    analysis.summary,
                    analysis.sentiment,
                    analysis.impact_magnitude,
                    analysis.event_type,
                    analysis.confidence,
                    analysis.estimated_price_move,
                    analysis.news_score,
                    analysis.analyzed_at.isoformat(),
                    analysis.metadata,
                    analysis.id,
                ),
            )
            await self.connection.commit()
            logger.debug(f"Updated analysis: {analysis.id}")
            
        except Exception as e:
            logger.error(f"Failed to update analysis {analysis.id}: {e}")
            raise AnalysisRepositoryError(f"Failed to update analysis: {e}") from e

    async def list_by_score_range(
        self,
        min_score: float,
        max_score: float,
        limit: int = 100,
    ) -> List[ArticleAnalysis]:
        """List analyses within a score range.
        
        Args:
            min_score: Minimum news score
            max_score: Maximum news score
            limit: Maximum number of records
            
        Returns:
            List of analyses
        """
        try:
            cursor = await self.connection.execute(
                """
                SELECT id, article_id, summary, sentiment, impact_magnitude, event_type,
                       confidence, estimated_price_move, news_score, analyzed_at, metadata
                FROM article_analysis
                WHERE news_score >= ? AND news_score <= ?
                ORDER BY analyzed_at DESC
                LIMIT ?
                """,
                (min_score, max_score, limit),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            
            return [self._row_to_analysis(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to list analyses by score range: {e}")
            raise AnalysisRepositoryError(f"Failed to list analyses: {e}") from e

    def _row_to_analysis(self, row: tuple) -> ArticleAnalysis:  # type: ignore[type-arg]
        """Convert database row to ArticleAnalysis object.
        
        Args:
            row: Database row tuple
            
        Returns:
            ArticleAnalysis object
        """
        return ArticleAnalysis(
            id=row[0],
            article_id=row[1],
            summary=row[2],
            sentiment=row[3],
            impact_magnitude=row[4],
            event_type=row[5],
            confidence=row[6],
            estimated_price_move=row[7],
            news_score=row[8],
            analyzed_at=datetime.fromisoformat(row[9]),
            metadata=row[10],
        )
