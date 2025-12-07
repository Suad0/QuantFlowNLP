"""Base repository interfaces and abstract classes.

This module defines the abstract base classes for all repositories,
providing common CRUD operations and establishing the repository pattern.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Generic, List, Optional, TypeVar

from app.models.domain import Article, ArticleAnalysis, OHLCVData, ScalerParams

# Type variable for generic repository
T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository with common CRUD operations.
    
    This class defines the interface that all repositories should implement
    for basic create, read, update, and delete operations.
    """

    @abstractmethod
    async def create(self, entity: T) -> str:
        """Create a new entity.
        
        Args:
            entity: Entity to create
            
        Returns:
            ID of created entity
        """
        pass

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """Retrieve entity by ID.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Entity if found, None otherwise
        """
        pass

    @abstractmethod
    async def update(self, entity: T) -> None:
        """Update an existing entity.
        
        Args:
            entity: Entity with updated values
        """
        pass

    @abstractmethod
    async def delete(self, entity_id: str) -> None:
        """Delete an entity.
        
        Args:
            entity_id: Entity identifier
        """
        pass


class ArticleRepository(ABC):
    """Repository interface for article operations.
    
    Defines all operations related to storing and retrieving news articles.
    """

    @abstractmethod
    async def create(self, article: Article) -> str:
        """Store a new article.
        
        Args:
            article: Article to store
            
        Returns:
            Article ID
        """
        pass

    @abstractmethod
    async def get_by_id(self, article_id: str) -> Optional[Article]:
        """Retrieve article by ID.
        
        Args:
            article_id: Article identifier
            
        Returns:
            Article if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_url(self, url: str) -> Optional[Article]:
        """Retrieve article by URL.
        
        Args:
            url: Article URL
            
        Returns:
            Article if found, None otherwise
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def get_unanalyzed(self, limit: int = 100) -> List[Article]:
        """Get articles that haven't been analyzed yet.
        
        Args:
            limit: Maximum number of articles to return
            
        Returns:
            List of unanalyzed articles
        """
        pass

    @abstractmethod
    async def count(self, source: Optional[str] = None) -> int:
        """Count total articles.
        
        Args:
            source: Optional source filter
            
        Returns:
            Total count
        """
        pass


class AnalysisRepository(ABC):
    """Repository interface for article analysis operations.
    
    Defines all operations related to storing and retrieving article analysis.
    """

    @abstractmethod
    async def create(self, analysis: ArticleAnalysis) -> str:
        """Store article analysis.
        
        Args:
            analysis: Analysis to store
            
        Returns:
            Analysis ID
        """
        pass

    @abstractmethod
    async def get_by_id(self, analysis_id: str) -> Optional[ArticleAnalysis]:
        """Retrieve analysis by ID.
        
        Args:
            analysis_id: Analysis identifier
            
        Returns:
            Analysis if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_article_id(self, article_id: str) -> Optional[ArticleAnalysis]:
        """Retrieve analysis for a specific article.
        
        Args:
            article_id: Article identifier
            
        Returns:
            Analysis if found, None otherwise
        """
        pass

    @abstractmethod
    async def update(self, analysis: ArticleAnalysis) -> None:
        """Update existing analysis.
        
        Args:
            analysis: Analysis with updated values
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def get_news_scores_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[tuple[datetime, float]]:
        """Get news scores within a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of (timestamp, news_score) tuples
        """
        pass


class OHLCVRepository(ABC):
    """Repository interface for OHLCV market data operations.
    
    Defines all operations related to storing and retrieving market data.
    """

    @abstractmethod
    async def insert(self, data: OHLCVData) -> str:
        """Insert OHLCV data point.
        
        Args:
            data: OHLCV data to insert
            
        Returns:
            Data ID
        """
        pass

    @abstractmethod
    async def insert_batch(self, data_list: List[OHLCVData]) -> int:
        """Insert multiple OHLCV data points.
        
        Args:
            data_list: List of OHLCV data
            
        Returns:
            Number of records inserted
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def get_latest(self, symbol: str, limit: int = 1) -> List[OHLCVData]:
        """Get latest OHLCV data points.
        
        Args:
            symbol: Trading symbol
            limit: Number of latest points to retrieve
            
        Returns:
            List of latest OHLCV data points
        """
        pass

    @abstractmethod
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
        pass


class ScalerRepository(ABC):
    """Repository interface for scaler parameters operations.
    
    Defines all operations related to storing and retrieving normalization parameters.
    """

    @abstractmethod
    async def save_params(self, params: ScalerParams) -> str:
        """Save scaler parameters.
        
        Args:
            params: Scaler parameters to save
            
        Returns:
            Parameters ID
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def get_all_params(self, symbol: str) -> List[ScalerParams]:
        """Get all scaler parameters for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of scaler parameters
        """
        pass

    @abstractmethod
    async def update_params(self, params: ScalerParams) -> None:
        """Update existing scaler parameters.
        
        Args:
            params: Updated scaler parameters
        """
        pass

    @abstractmethod
    async def delete_params(self, symbol: str, feature_name: str) -> None:
        """Delete scaler parameters.
        
        Args:
            symbol: Trading symbol
            feature_name: Feature name
        """
        pass


class VectorRepository(ABC):
    """Repository interface for vector embedding operations.
    
    Defines all operations related to storing and querying article embeddings
    in the vector database for similarity search and deduplication.
    """

    @abstractmethod
    async def store_embedding(
        self,
        article_id: str,
        embedding: List[float],
        metadata: dict,
    ) -> None:
        """Store an article embedding with metadata.
        
        Args:
            article_id: Unique article identifier
            embedding: Embedding vector
            metadata: Additional metadata (source, title, published_at, etc.)
        """
        pass

    @abstractmethod
    async def find_similar(
        self,
        embedding: List[float],
        threshold: float = 0.95,
        limit: int = 10,
    ) -> List[dict]:
        """Find similar embeddings above a similarity threshold.
        
        Args:
            embedding: Query embedding vector
            threshold: Minimum similarity threshold (0-1)
            limit: Maximum number of results
            
        Returns:
            List of similar articles with similarity scores and metadata
        """
        pass

    @abstractmethod
    async def get_by_article_id(self, article_id: str) -> Optional[dict]:
        """Get embedding by article ID.
        
        Args:
            article_id: Article identifier
            
        Returns:
            Embedding data with metadata if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete_by_article_id(self, article_id: str) -> None:
        """Delete embedding by article ID.
        
        Args:
            article_id: Article identifier
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get total count of stored embeddings.
        
        Returns:
            Number of embeddings in storage
        """
        pass
