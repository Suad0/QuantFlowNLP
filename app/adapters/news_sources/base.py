"""Base news source adapter interface.

This module defines the abstract base class for all news source adapters,
providing a consistent interface for fetching articles from different sources.
"""

from abc import ABC, abstractmethod

from app.models.domain.article import RawArticle


class NewsSourceAdapter(ABC):
    """Abstract base class for news source adapters.
    
    All news source implementations must inherit from this class and implement
    the fetch method to retrieve articles from their respective sources.
    
    Attributes:
        source_name: Identifier for the news source (e.g., "yahoo", "reuters")
        timeout: Request timeout in seconds
        max_articles: Maximum number of articles to fetch
    """
    
    def __init__(
        self,
        source_name: str,
        timeout: int = 30,
        max_articles: int = 100,
    ) -> None:
        """Initialize the news source adapter.
        
        Args:
            source_name: Identifier for the news source
            timeout: Request timeout in seconds
            max_articles: Maximum number of articles to fetch
        """
        self.source_name = source_name
        self.timeout = timeout
        self.max_articles = max_articles
    
    @abstractmethod
    async def fetch(self) -> list[RawArticle]:
        """Fetch articles from the news source.
        
        This method must be implemented by all concrete news source adapters.
        It should fetch articles from the source, parse them, and return a list
        of RawArticle objects.
        
        Returns:
            List of RawArticle objects fetched from the source
            
        Raises:
            NewsIngestionError: If there are issues fetching or parsing articles
        """
        pass
