"""Investing.com API adapter.

This module implements the news source adapter for Investing.com API,
fetching financial news articles with API authentication and rate limiting.
"""

import asyncio
from datetime import datetime
from typing import Any, Optional

import httpx

from app.adapters.news_sources.base import NewsSourceAdapter
from app.core.exceptions import NewsIngestionError
from app.models.domain.article import RawArticle


class InvestingComAdapter(NewsSourceAdapter):
    """Adapter for fetching articles from Investing.com API.
    
    This adapter handles API authentication, rate limiting, and error responses
    from the Investing.com news API.
    
    Note: Investing.com requires API credentials. Set INVESTING_COM_API_KEY
    environment variable or pass api_key parameter.
    """
    
    API_BASE_URL = "https://api.investing.com/api/financialdata"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_articles: int = 100,
        rate_limit_delay: float = 1.0,
    ) -> None:
        """Initialize Investing.com adapter.
        
        Args:
            api_key: API key for authentication (optional, can use env var)
            timeout: Request timeout in seconds
            max_articles: Maximum number of articles to fetch
            rate_limit_delay: Delay between requests in seconds
        """
        super().__init__(
            source_name="investing_com",
            timeout=timeout,
            max_articles=max_articles,
        )
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: Optional[float] = None
    
    async def fetch(self) -> list[RawArticle]:
        """Fetch articles from Investing.com API.
        
        Returns:
            List of RawArticle objects parsed from the API response
            
        Raises:
            NewsIngestionError: If fetching or parsing fails
        """
        if not self.api_key:
            # If no API key provided, return empty list instead of failing
            # This allows the system to continue with other sources
            return []
        
        try:
            articles: list[RawArticle] = []
            
            # Fetch articles with pagination
            page = 1
            while len(articles) < self.max_articles:
                await self._rate_limit()
                
                page_articles = await self._fetch_page(page)
                if not page_articles:
                    break
                
                articles.extend(page_articles)
                page += 1
                
                # Stop if we've reached max articles
                if len(articles) >= self.max_articles:
                    articles = articles[:self.max_articles]
                    break
            
            return articles
            
        except httpx.HTTPError as e:
            raise NewsIngestionError(
                f"Failed to fetch from Investing.com API: {e}"
            ) from e
    
    async def _fetch_page(self, page: int) -> list[RawArticle]:
        """Fetch a single page of articles from the API.
        
        Args:
            page: Page number to fetch
            
        Returns:
            List of RawArticle objects from this page
            
        Raises:
            NewsIngestionError: If API request fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        params = {
            "page": page,
            "limit": min(50, self.max_articles),  # API typically limits to 50 per page
            "category": "news",
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    f"{self.API_BASE_URL}/news",
                    headers=headers,
                    params=params,
                )
                
                # Handle rate limiting (429 status)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", "60"))
                    await asyncio.sleep(retry_after)
                    # Retry the request
                    response = await client.get(
                        f"{self.API_BASE_URL}/news",
                        headers=headers,
                        params=params,
                    )
                
                response.raise_for_status()
                data = response.json()
                
                return self._parse_response(data)
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise NewsIngestionError(
                        "Invalid Investing.com API key"
                    ) from e
                elif e.response.status_code == 403:
                    raise NewsIngestionError(
                        "Access forbidden - check API key permissions"
                    ) from e
                else:
                    raise NewsIngestionError(
                        f"Investing.com API error: {e.response.status_code}"
                    ) from e
    
    def _parse_response(self, data: dict[str, Any]) -> list[RawArticle]:
        """Parse API response into RawArticle objects.
        
        Args:
            data: JSON response from the API
            
        Returns:
            List of parsed RawArticle objects
        """
        articles: list[RawArticle] = []
        
        # API response structure may vary, adjust as needed
        items = data.get("data", []) or data.get("articles", [])
        
        for item in items:
            article = self._parse_item(item)
            if article:
                articles.append(article)
        
        return articles
    
    def _parse_item(self, item: dict[str, Any]) -> Optional[RawArticle]:
        """Parse a single API item into a RawArticle.
        
        Args:
            item: Dictionary representing an article from the API
            
        Returns:
            RawArticle object or None if parsing fails
        """
        try:
            title = item.get("title", "")
            url = item.get("url", "") or item.get("link", "")
            
            # Get content, fallback to description or summary
            content = (
                item.get("content", "")
                or item.get("description", "")
                or item.get("summary", "")
            )
            
            if not all([title, url]):
                return None
            
            # Parse publication date
            published_at = self._parse_date(
                item.get("published_at") or item.get("publishedAt") or item.get("date")
            )
            
            # Extract additional metadata
            metadata = {
                "feed": "investing_com_api",
                "category": item.get("category"),
                "tags": item.get("tags", []),
            }
            
            return RawArticle(
                title=title,
                content=content,
                source=self.source_name,
                url=url,
                published_at=published_at,
                metadata=metadata,
            )
            
        except Exception:
            # Skip items that fail to parse
            return None
    
    def _parse_date(self, date_value: Any) -> datetime:
        """Parse date from various formats.
        
        Args:
            date_value: Date as string, timestamp, or datetime
            
        Returns:
            Parsed datetime object, or current time if parsing fails
        """
        if not date_value:
            return datetime.now()
        
        try:
            # If it's already a datetime
            if isinstance(date_value, datetime):
                return date_value
            
            # If it's a Unix timestamp
            if isinstance(date_value, (int, float)):
                return datetime.fromtimestamp(date_value)
            
            # If it's an ISO format string
            if isinstance(date_value, str):
                # Try ISO format
                try:
                    return datetime.fromisoformat(date_value.replace("Z", "+00:00"))
                except ValueError:
                    pass
                
                # Try other common formats
                from dateutil import parser
                return parser.parse(date_value)
            
        except Exception:
            pass
        
        return datetime.now()
    
    async def _rate_limit(self) -> None:
        """Implement rate limiting between requests.
        
        Ensures minimum delay between consecutive API requests to avoid
        hitting rate limits.
        """
        if self._last_request_time is not None:
            elapsed = asyncio.get_event_loop().time() - self._last_request_time
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
        
        self._last_request_time = asyncio.get_event_loop().time()
