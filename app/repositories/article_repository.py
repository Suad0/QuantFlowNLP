"""SQLite implementation of ArticleRepository.

This module provides the concrete implementation of article storage
using SQLite with aiosqlite for async operations.
"""

import aiosqlite
from datetime import datetime
from typing import List, Optional

from app.core.exceptions import ArticleRepositoryError
from app.models.domain import Article
from app.repositories.base import ArticleRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteArticleRepository(ArticleRepository):
    """SQLite implementation of ArticleRepository.
    
    Provides async operations for storing and retrieving articles
    from SQLite database.
    """

    def __init__(self, connection: aiosqlite.Connection):
        """Initialize repository with database connection.
        
        Args:
            connection: Active aiosqlite connection
        """
        self.connection = connection

    async def create(self, article: Article) -> str:
        """Store a new article.
        
        Args:
            article: Article to store
            
        Returns:
            Article ID
            
        Raises:
            ArticleRepositoryError: If creation fails
        """
        try:
            await self.connection.execute(
                """
                INSERT INTO articles 
                (id, title, content, summary, source, url, published_at, fetched_at, 
                 is_duplicate, duplicate_of, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article.id,
                    article.title,
                    article.content,
                    article.summary,
                    article.source,
                    article.url,
                    article.published_at.isoformat(),
                    article.fetched_at.isoformat(),
                    article.is_duplicate,
                    article.duplicate_of,
                    article.metadata,
                ),
            )
            await self.connection.commit()
            logger.debug(f"Created article: {article.id}")
            return article.id
            
        except aiosqlite.IntegrityError as e:
            logger.error(f"Article already exists or constraint violation: {e}")
            raise ArticleRepositoryError(f"Failed to create article: {e}") from e
        except Exception as e:
            logger.error(f"Failed to create article: {e}")
            raise ArticleRepositoryError(f"Failed to create article: {e}") from e

    async def get_by_id(self, article_id: str) -> Optional[Article]:
        """Retrieve article by ID.
        
        Args:
            article_id: Article identifier
            
        Returns:
            Article if found, None otherwise
        """
        try:
            cursor = await self.connection.execute(
                """
                SELECT id, title, content, summary, source, url, published_at, 
                       fetched_at, is_duplicate, duplicate_of, metadata
                FROM articles
                WHERE id = ?
                """,
                (article_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            
            if row:
                return self._row_to_article(row)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get article {article_id}: {e}")
            raise ArticleRepositoryError(f"Failed to get article: {e}") from e

    async def get_by_url(self, url: str) -> Optional[Article]:
        """Retrieve article by URL.
        
        Args:
            url: Article URL
            
        Returns:
            Article if found, None otherwise
        """
        try:
            cursor = await self.connection.execute(
                """
                SELECT id, title, content, summary, source, url, published_at, 
                       fetched_at, is_duplicate, duplicate_of, metadata
                FROM articles
                WHERE url = ?
                """,
                (url,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            
            if row:
                return self._row_to_article(row)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get article by URL {url}: {e}")
            raise ArticleRepositoryError(f"Failed to get article by URL: {e}") from e

    async def list_articles(
        self,
        skip: int = 0,
        limit: int = 50,
        source: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Article]:
        """List articles with pagination and filtering.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            source: Optional source filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of articles
        """
        try:
            query = """
                SELECT id, title, content, summary, source, url, published_at, 
                       fetched_at, is_duplicate, duplicate_of, metadata
                FROM articles
                WHERE 1=1
            """
            params: List[str | int] = []
            
            if source:
                query += " AND source = ?"
                params.append(source)
            
            if start_date:
                query += " AND published_at >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND published_at <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY published_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, skip])
            
            cursor = await self.connection.execute(query, params)
            rows = await cursor.fetchall()
            await cursor.close()
            
            return [self._row_to_article(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to list articles: {e}")
            raise ArticleRepositoryError(f"Failed to list articles: {e}") from e

    async def get_unanalyzed(self, limit: int = 100) -> List[Article]:
        """Get articles that haven't been analyzed yet.
        
        Args:
            limit: Maximum number of articles to return
            
        Returns:
            List of unanalyzed articles
        """
        try:
            cursor = await self.connection.execute(
                """
                SELECT a.id, a.title, a.content, a.summary, a.source, a.url, 
                       a.published_at, a.fetched_at, a.is_duplicate, a.duplicate_of, a.metadata
                FROM articles a
                LEFT JOIN article_analysis aa ON a.id = aa.article_id
                WHERE aa.id IS NULL AND a.is_duplicate = FALSE
                ORDER BY a.published_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            
            return [self._row_to_article(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get unanalyzed articles: {e}")
            raise ArticleRepositoryError(f"Failed to get unanalyzed articles: {e}") from e

    async def count(self, source: Optional[str] = None) -> int:
        """Count total articles.
        
        Args:
            source: Optional source filter
            
        Returns:
            Total count
        """
        try:
            if source:
                cursor = await self.connection.execute(
                    "SELECT COUNT(*) FROM articles WHERE source = ?",
                    (source,),
                )
            else:
                cursor = await self.connection.execute("SELECT COUNT(*) FROM articles")
            
            row = await cursor.fetchone()
            await cursor.close()
            
            return row[0] if row else 0
            
        except Exception as e:
            logger.error(f"Failed to count articles: {e}")
            raise ArticleRepositoryError(f"Failed to count articles: {e}") from e

    def _row_to_article(self, row: tuple) -> Article:  # type: ignore[type-arg]
        """Convert database row to Article object.
        
        Args:
            row: Database row tuple
            
        Returns:
            Article object
        """
        return Article(
            id=row[0],
            title=row[1],
            content=row[2],
            summary=row[3],
            source=row[4],
            url=row[5],
            published_at=datetime.fromisoformat(row[6]),
            fetched_at=datetime.fromisoformat(row[7]),
            is_duplicate=bool(row[8]),
            duplicate_of=row[9],
            metadata=row[10],
        )
