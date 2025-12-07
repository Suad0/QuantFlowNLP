"""Unit tests for NewsIngestionService.

This module tests the news ingestion service functionality including
article fetching, normalization, embedding generation, and deduplication.
"""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import NewsIngestionError
from app.models.domain.article import Article, RawArticle
from app.services.news_ingestion import NewsIngestionService


@pytest.fixture
def mock_article_repo():
    """Create mock article repository."""
    repo = AsyncMock()
    repo.create = AsyncMock(return_value="test-article-id")
    repo.get_by_url = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_vector_repo():
    """Create mock vector repository."""
    repo = AsyncMock()
    repo.store_embedding = AsyncMock()
    repo.check_duplicate = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_ollama_client():
    """Create mock Ollama client."""
    client = AsyncMock()
    client.embed = AsyncMock(return_value=[0.1] * 768)
    return client


@pytest.fixture
def mock_news_source():
    """Create mock news source adapter."""
    source = MagicMock()
    source.source_name = "test_source"
    source.fetch = AsyncMock(return_value=[
        RawArticle(
            title="Test Article",
            content="Test content",
            source="test_source",
            url="https://example.com/article1",
            published_at=datetime.now(),
            metadata={"author": "Test Author"},
        )
    ])
    return source


@pytest.fixture
def news_ingestion_service(
    mock_article_repo,
    mock_vector_repo,
    mock_ollama_client,
    mock_news_source,
):
    """Create NewsIngestionService with mocked dependencies."""
    return NewsIngestionService(
        article_repo=mock_article_repo,
        vector_repo=mock_vector_repo,
        ollama_client=mock_ollama_client,
        news_sources=[mock_news_source],
    )


@pytest.mark.asyncio
async def test_normalize_article():
    """Test article normalization."""
    service = NewsIngestionService(
        article_repo=AsyncMock(),
        vector_repo=AsyncMock(),
        ollama_client=AsyncMock(),
        news_sources=[],
    )
    
    raw = RawArticle(
        title="  Test Title  ",
        content="  Test Content  ",
        source="test_source",
        url="https://example.com/test",
        published_at=datetime(2024, 1, 1, 12, 0, 0),
        metadata={"key": "value"},
    )
    
    article = await service.normalize_article(raw)
    
    assert article.title == "Test Title"
    assert article.content == "Test Content"
    assert article.source == "test_source"
    assert article.url == "https://example.com/test"
    assert article.published_at == datetime(2024, 1, 1, 12, 0, 0)
    assert article.is_duplicate is False
    assert article.duplicate_of is None
    assert article.metadata == "{'key': 'value'}"


@pytest.mark.asyncio
async def test_generate_embedding(news_ingestion_service, mock_ollama_client):
    """Test embedding generation."""
    embedding = await news_ingestion_service.generate_embedding("Test text")
    
    assert len(embedding) == 768
    assert all(isinstance(x, float) for x in embedding)
    mock_ollama_client.embed.assert_called_once()


@pytest.mark.asyncio
async def test_generate_embedding_failure(news_ingestion_service, mock_ollama_client):
    """Test embedding generation failure handling."""
    mock_ollama_client.embed.side_effect = Exception("Ollama error")
    
    with pytest.raises(NewsIngestionError, match="Failed to generate embedding"):
        await news_ingestion_service.generate_embedding("Test text")


@pytest.mark.asyncio
async def test_check_duplicate_found(news_ingestion_service, mock_vector_repo):
    """Test duplicate detection when duplicate exists."""
    mock_vector_repo.check_duplicate.return_value = "existing-article-id"
    
    embedding = [0.1] * 768
    duplicate_id = await news_ingestion_service.check_duplicate(embedding)
    
    assert duplicate_id == "existing-article-id"
    mock_vector_repo.check_duplicate.assert_called_once()


@pytest.mark.asyncio
async def test_check_duplicate_not_found(news_ingestion_service, mock_vector_repo):
    """Test duplicate detection when no duplicate exists."""
    mock_vector_repo.check_duplicate.return_value = None
    
    embedding = [0.1] * 768
    duplicate_id = await news_ingestion_service.check_duplicate(embedding)
    
    assert duplicate_id is None


@pytest.mark.asyncio
async def test_store_article(news_ingestion_service, mock_article_repo, mock_vector_repo):
    """Test article and embedding storage."""
    article = Article(
        id="test-id",
        title="Test",
        content="Content",
        source="test",
        url="https://example.com",
        published_at=datetime.now(),
        fetched_at=datetime.now(),
    )
    embedding = [0.1] * 768
    
    article_id = await news_ingestion_service.store_article(article, embedding)
    
    assert article_id == "test-article-id"
    mock_article_repo.create.assert_called_once_with(article)
    mock_vector_repo.store_embedding.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_articles(news_ingestion_service, mock_news_source):
    """Test fetching articles from a source."""
    articles = await news_ingestion_service.fetch_articles(mock_news_source)
    
    assert len(articles) == 1
    assert articles[0].title == "Test Article"
    mock_news_source.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_articles_with_limit(news_ingestion_service, mock_news_source):
    """Test fetching articles with limit."""
    # Mock source returns 5 articles
    mock_news_source.fetch.return_value = [
        RawArticle(
            title=f"Article {i}",
            content="Content",
            source="test",
            url=f"https://example.com/{i}",
            published_at=datetime.now(),
        )
        for i in range(5)
    ]
    
    articles = await news_ingestion_service.fetch_articles(mock_news_source, max_articles=3)
    
    assert len(articles) == 3


@pytest.mark.asyncio
async def test_ingest_from_all_sources_success(
    news_ingestion_service,
    mock_article_repo,
    mock_vector_repo,
    mock_ollama_client,
):
    """Test successful ingestion from all sources."""
    result = await news_ingestion_service.ingest_from_all_sources()
    
    assert result["total_fetched"] == 1
    assert result["total_stored"] == 1
    assert result["duplicates_found"] == 0
    assert len(result["errors"]) == 0
    assert "test_source" in result["by_source"]
    assert result["by_source"]["test_source"]["fetched"] == 1
    assert result["by_source"]["test_source"]["stored"] == 1


@pytest.mark.asyncio
async def test_ingest_from_all_sources_with_duplicate(
    news_ingestion_service,
    mock_vector_repo,
):
    """Test ingestion with duplicate detection."""
    mock_vector_repo.check_duplicate.return_value = "existing-id"
    
    result = await news_ingestion_service.ingest_from_all_sources()
    
    assert result["total_fetched"] == 1
    assert result["total_stored"] == 1
    assert result["duplicates_found"] == 1


@pytest.mark.asyncio
async def test_ingest_from_all_sources_with_existing_url(
    news_ingestion_service,
    mock_article_repo,
):
    """Test ingestion with existing URL."""
    mock_article_repo.get_by_url.return_value = Article(
        id="existing-id",
        title="Existing",
        content="Content",
        source="test",
        url="https://example.com/article1",
        published_at=datetime.now(),
        fetched_at=datetime.now(),
    )
    
    result = await news_ingestion_service.ingest_from_all_sources()
    
    assert result["total_fetched"] == 1
    assert result["total_stored"] == 0
    assert result["duplicates_found"] == 1


@pytest.mark.asyncio
async def test_ingest_from_all_sources_with_source_error(
    mock_article_repo,
    mock_vector_repo,
    mock_ollama_client,
):
    """Test ingestion with source fetch error."""
    failing_source = MagicMock()
    failing_source.source_name = "failing_source"
    failing_source.fetch = AsyncMock(side_effect=Exception("Fetch failed"))
    
    service = NewsIngestionService(
        article_repo=mock_article_repo,
        vector_repo=mock_vector_repo,
        ollama_client=mock_ollama_client,
        news_sources=[failing_source],
    )
    
    result = await service.ingest_from_all_sources()
    
    assert result["total_fetched"] == 0
    assert result["total_stored"] == 0
    assert len(result["errors"]) == 1
    assert "failing_source" in result["errors"][0]
    assert "failing_source" in result["by_source"]
    assert result["by_source"]["failing_source"]["error"] is not None
