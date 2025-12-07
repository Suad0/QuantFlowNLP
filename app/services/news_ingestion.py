"""News ingestion service for fetching and storing financial news articles.

This module provides the NewsIngestionService which orchestrates the process of
fetching articles from multiple news sources, normalizing them, generating embeddings,
checking for duplicates, and storing them in the database.
"""

import asyncio
import uuid
from datetime import datetime

from app.adapters.news_sources.base import NewsSourceAdapter
from app.adapters.ollama_client import OllamaClient
from app.core.config import settings
from app.core.exceptions import NewsIngestionError
from app.models.domain.article import Article, RawArticle
from app.repositories.article_repository import SQLiteArticleRepository
from app.repositories.vector_repository import ChromaVectorRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class NewsIngestionService:
    """Service for ingesting financial news from multiple sources.

    This service orchestrates the complete news ingestion pipeline:
    1. Fetch articles from multiple news sources
    2. Normalize articles to unified schema
    3. Generate embeddings for deduplication
    4. Check for duplicates using similarity matching
    5. Store articles and embeddings

    Attributes:
        article_repo: Repository for article storage
        vector_repo: Repository for embedding storage
        ollama_client: Client for embedding generation
        news_sources: List of news source adapters
    """

    def __init__(
        self,
        article_repo: SQLiteArticleRepository,
        vector_repo: ChromaVectorRepository,
        ollama_client: OllamaClient,
        news_sources: list[NewsSourceAdapter],
    ):
        """Initialize the news ingestion service.

        Args:
            article_repo: Repository for article storage
            vector_repo: Repository for embedding storage
            ollama_client: Client for embedding generation
            news_sources: List of news source adapters to fetch from
        """
        self.article_repo = article_repo
        self.vector_repo = vector_repo
        self.ollama_client = ollama_client
        self.news_sources = news_sources

        logger.info(
            f"Initialized NewsIngestionService with {len(news_sources)} sources"
        )

    async def ingest_from_all_sources(
        self,
        max_articles_per_source: int | None = None,
    ) -> dict:
        """Ingest articles from all configured news sources.

        This method orchestrates concurrent fetching from all news sources,
        processes each article through the ingestion pipeline, and returns
        a summary of the ingestion results.

        Args:
            max_articles_per_source: Optional limit on articles per source

        Returns:
            Dictionary containing ingestion statistics:
                - total_fetched: Total articles fetched
                - total_stored: Total articles successfully stored
                - duplicates_found: Number of duplicates detected
                - errors: List of error messages
                - duration_seconds: Total processing time
                - by_source: Per-source statistics
        """
        start_time = datetime.now()

        logger.info(
            f"Starting ingestion from {len(self.news_sources)} sources",
            extra={"max_articles_per_source": max_articles_per_source},
        )

        # Statistics tracking
        total_fetched = 0
        total_stored = 0
        duplicates_found = 0
        errors: list[str] = []
        by_source: dict = {}

        # Fetch from all sources concurrently
        fetch_tasks = [
            self.fetch_articles(source, max_articles_per_source)
            for source in self.news_sources
        ]

        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        # Process results from each source
        for source, result in zip(self.news_sources, fetch_results, strict=False):
            source_name = source.source_name

            if isinstance(result, Exception):
                error_msg = f"Failed to fetch from {source_name}: {str(result)}"
                logger.error(error_msg)
                errors.append(error_msg)
                by_source[source_name] = {
                    "fetched": 0,
                    "stored": 0,
                    "duplicates": 0,
                    "error": str(result),
                }
                continue

            # Process articles from this source
            raw_articles: list[RawArticle] = result
            source_fetched = len(raw_articles)
            source_stored = 0
            source_duplicates = 0

            logger.info(f"Fetched {source_fetched} articles from {source_name}")

            for raw_article in raw_articles:
                try:
                    # Normalize article
                    article = await self.normalize_article(raw_article)

                    # Check if URL already exists
                    existing = await self.article_repo.get_by_url(article.url)
                    if existing:
                        logger.debug(f"Article already exists: {article.url}")
                        source_duplicates += 1
                        continue

                    # Generate embedding
                    embedding = await self.generate_embedding(
                        f"{article.title} {article.content}"
                    )

                    # Check for semantic duplicates
                    duplicate_id = await self.check_duplicate(embedding)

                    if duplicate_id:
                        # Mark as duplicate and store
                        article.is_duplicate = True
                        article.duplicate_of = duplicate_id
                        source_duplicates += 1
                        logger.info(
                            f"Duplicate detected: {article.id} -> {duplicate_id}"
                        )

                    # Store article and embedding
                    await self.store_article(article, embedding)
                    source_stored += 1

                except Exception as e:
                    error_msg = f"Failed to process article from {source_name}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    errors.append(error_msg)

            # Update statistics
            total_fetched += source_fetched
            total_stored += source_stored
            duplicates_found += source_duplicates

            by_source[source_name] = {
                "fetched": source_fetched,
                "stored": source_stored,
                "duplicates": source_duplicates,
            }

            logger.info(
                f"Completed {source_name}: "
                f"fetched={source_fetched}, stored={source_stored}, "
                f"duplicates={source_duplicates}"
            )

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()

        result = {
            "total_fetched": total_fetched,
            "total_stored": total_stored,
            "duplicates_found": duplicates_found,
            "errors": errors,
            "duration_seconds": duration,
            "by_source": by_source,
        }

        logger.info(
            f"Ingestion complete: {total_stored}/{total_fetched} articles stored, "
            f"{duplicates_found} duplicates, {len(errors)} errors, "
            f"{duration:.2f}s"
        )

        return result

    async def fetch_articles(
        self,
        source: NewsSourceAdapter,
        max_articles: int | None = None,
    ) -> list[RawArticle]:
        """Fetch articles from a single news source with timeout handling.

        Args:
            source: News source adapter to fetch from
            max_articles: Optional limit on number of articles

        Returns:
            List of raw articles fetched from the source

        Raises:
            NewsIngestionError: If fetching fails or times out
        """
        source_name = source.source_name
        timeout = settings.news_fetch_timeout

        logger.debug(
            f"Fetching from {source_name}",
            extra={"timeout": timeout, "max_articles": max_articles},
        )

        try:
            # Fetch with timeout
            articles = await asyncio.wait_for(
                source.fetch(),
                timeout=timeout,
            )

            # Apply limit if specified
            if max_articles and len(articles) > max_articles:
                articles = articles[:max_articles]
                logger.debug(
                    f"Limited {source_name} to {max_articles} articles"
                )

            logger.info(f"Fetched {len(articles)} articles from {source_name}")
            return articles

        except TimeoutError as e:
            error_msg = f"Timeout fetching from {source_name} after {timeout}s"
            logger.error(error_msg)
            raise NewsIngestionError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to fetch from {source_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise NewsIngestionError(error_msg) from e

    async def normalize_article(self, raw: RawArticle) -> Article:
        """Normalize a raw article into the unified Article schema.

        This method converts a RawArticle (from various sources) into the
        standardized Article domain model with consistent fields and types.

        Args:
            raw: Raw article from news source

        Returns:
            Normalized Article object
        """
        # Generate unique ID
        article_id = str(uuid.uuid4())

        # Current timestamp for fetched_at
        fetched_at = datetime.now()

        # Create normalized article
        article = Article(
            id=article_id,
            title=raw.title.strip(),
            content=raw.content.strip(),
            source=raw.source,
            url=raw.url,
            published_at=raw.published_at,
            fetched_at=fetched_at,
            summary=None,  # Will be generated by NLP service
            is_duplicate=False,
            duplicate_of=None,
            metadata=str(raw.metadata) if raw.metadata else None,
        )

        logger.debug(
            f"Normalized article: {article_id}",
            extra={
                "source": article.source,
                "title": article.title[:50],
                "published_at": article.published_at.isoformat(),
            },
        )

        return article

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using Ollama.

        Args:
            text: Text to generate embedding for

        Returns:
            Embedding vector as list of floats

        Raises:
            NewsIngestionError: If embedding generation fails
        """
        try:
            embedding = await self.ollama_client.embed(
                model=settings.ollama_embedding_model,
                text=text,
            )

            logger.debug(f"Generated embedding with dimension {len(embedding)}")
            return embedding

        except Exception as e:
            error_msg = f"Failed to generate embedding: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise NewsIngestionError(error_msg) from e

    async def check_duplicate(
        self,
        embedding: list[float],
        threshold: float | None = None,
    ) -> str | None:
        """Check if an article is a duplicate using similarity matching.

        Args:
            embedding: Article embedding vector
            threshold: Similarity threshold (default from settings)

        Returns:
            Article ID of duplicate if found, None otherwise
        """
        _threshold = threshold or settings.news_deduplication_threshold

        try:
            duplicate_id = await self.vector_repo.check_duplicate(
                embedding=embedding,
                threshold=_threshold,
            )

            if duplicate_id:
                logger.info(f"Duplicate found: {duplicate_id}")

            return duplicate_id

        except Exception as e:
            # Log error but don't fail ingestion
            logger.error(f"Error checking for duplicates: {e}", exc_info=True)
            return None

    async def store_article(
        self,
        article: Article,
        embedding: list[float],
    ) -> str:
        """Store article and its embedding.

        Args:
            article: Article to store
            embedding: Article embedding vector

        Returns:
            Article ID

        Raises:
            NewsIngestionError: If storage fails
        """
        try:
            # Store article in database
            article_id = await self.article_repo.create(article)

            # Store embedding in vector database
            metadata = {
                "article_id": article_id,
                "source": article.source,
                "title": article.title,
                "published_at": article.published_at,
                "url": article.url,
            }

            await self.vector_repo.store_embedding(
                article_id=article_id,
                embedding=embedding,
                metadata=metadata,
            )

            logger.debug(f"Stored article and embedding: {article_id}")
            return article_id

        except Exception as e:
            error_msg = f"Failed to store article {article.id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise NewsIngestionError(error_msg) from e
