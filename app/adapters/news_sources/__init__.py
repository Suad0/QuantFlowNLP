"""News source adapters."""

from app.adapters.news_sources.base import NewsSourceAdapter
from app.adapters.news_sources.bloomberg import BloombergAdapter
from app.adapters.news_sources.investing_com import InvestingComAdapter
from app.adapters.news_sources.marketwatch import MarketWatchAdapter
from app.adapters.news_sources.reuters import ReutersAdapter
from app.adapters.news_sources.yahoo_finance import YahooFinanceAdapter

__all__ = [
    "NewsSourceAdapter",
    "YahooFinanceAdapter",
    "ReutersAdapter",
    "BloombergAdapter",
    "MarketWatchAdapter",
    "InvestingComAdapter",
]
