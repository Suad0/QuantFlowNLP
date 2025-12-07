"""Yahoo Finance RSS feed adapter.

This module implements the news source adapter for Yahoo Finance RSS feeds,
fetching financial news articles and parsing them into RawArticle objects.
"""

import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional

import httpx

from app.adapters.news_sources.base import NewsSourceAdapter
from app.core.exceptions import NewsIngestionError
from app.models.domain.article import RawArticle


class YahooFinanceAdapter(NewsSourceAdapter):
    """Adapter for fetching articles from Yahoo Finance RSS feeds.
    
    Yahoo Finance provides RSS feeds for various financial news categories.
    This adapter fetches articles from the main finance RSS feed.
    """
    
    RSS_URL = "https://finance.yahoo.com/news/rssindex"
    
    def __init__(self, timeout: int = 30, max_articles: int = 100) -> None:
        """Initialize Yahoo Finance adapter.
        
        Args:
            timeout: Request timeout in seconds
            max_articles: Maximum number of articles to fetch
        """
        super().__init__(
            source_name="yahoo",
            timeout=timeout,
            max_articles=max_articles,
        )
    
    async def fetch(self) -> list[RawArticle]:
        """Fetch articles from Yahoo Finance RSS feed.
        
        Returns:
            List of RawArticle objects parsed from the RSS feed
            
        Raises:
            NewsIngestionError: If fetching or parsing fails
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.RSS_URL)
                response.raise_for_status()
                
                return self._parse_rss(response.text)
                
        except httpx.HTTPError as e:
            raise NewsIngestionError(
                f"Failed to fetch Yahoo Finance RSS feed: {e}"
            ) from e
        except ET.ParseError as e:
            raise NewsIngestionError(
                f"Failed to parse Yahoo Finance RSS feed: {e}"
            ) from e
    
    def _parse_rss(self, rss_content: str) -> list[RawArticle]:
        """Parse RSS feed content into RawArticle objects.
        
        Args:
            rss_content: Raw RSS XML content
            
        Returns:
            List of parsed RawArticle objects
        """
        articles: list[RawArticle] = []
        
        try:
            root = ET.fromstring(rss_content)
            
            # Find all item elements in the RSS feed
            items = root.findall(".//item")[:self.max_articles]
            
            for item in items:
                article = self._parse_item(item)
                if article:
                    articles.append(article)
                    
        except Exception as e:
            raise NewsIngestionError(
                f"Error parsing Yahoo Finance RSS items: {e}"
            ) from e
        
        return articles
    
    def _parse_item(self, item: ET.Element) -> Optional[RawArticle]:
        """Parse a single RSS item into a RawArticle.
        
        Args:
            item: XML element representing an RSS item
            
        Returns:
            RawArticle object or None if parsing fails
        """
        try:
            title_elem = item.find("title")
            link_elem = item.find("link")
            description_elem = item.find("description")
            pub_date_elem = item.find("pubDate")
            
            if not all([title_elem, link_elem]):
                return None
            
            title = title_elem.text or ""
            url = link_elem.text or ""
            
            # Use description as content (fallback to summary)
            content = description_elem.text if description_elem is not None else ""
            
            # Parse publication date
            published_at = self._parse_date(
                pub_date_elem.text if pub_date_elem is not None else None
            )
            
            return RawArticle(
                title=title,
                content=content,
                source=self.source_name,
                url=url,
                published_at=published_at,
                metadata={"feed": "yahoo_finance_rss"},
            )
            
        except Exception:
            # Skip items that fail to parse
            return None
    
    def _parse_date(self, date_str: Optional[str]) -> datetime:
        """Parse RSS date string into datetime object.
        
        Args:
            date_str: Date string in RFC 822 format (e.g., "Mon, 01 Jan 2024 12:00:00 GMT")
            
        Returns:
            Parsed datetime object, or current time if parsing fails
        """
        if not date_str:
            return datetime.now()
        
        try:
            # Try parsing RFC 822 format (common in RSS)
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(date_str)
        except Exception:
            # Fallback to current time if parsing fails
            return datetime.now()
