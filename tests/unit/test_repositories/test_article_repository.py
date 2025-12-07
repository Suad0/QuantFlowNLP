"""Unit tests for ArticleRepository.

Tests the SQLite implementation of article storage and retrieval.
"""

import pytest
import aiosqlite
from datetime import datetime, timezone
import uuid

from app.repositories import SQLiteArticleRepository
from app.models.domain import Article
from app.adapters.database import DatabaseManager


@pytest.fixture
async def db_connection():
    """Create in-memory database connection for testing."""
    db_manager = DatabaseManager(":memory:")
    conn = await db_manager.connect()
    await db_manager.initialize_schema()
    yield conn
    await db_manager.disconnect()


@pytest.fixture
def sample_article():
    """Create a sample article for testing."""
    return Article(
        id=str(uuid.uuid4()),
        title="Test Article",
        content="This is test content for the article.",
        source="reuters",
        url=f"https://example.com/article/{uuid.uuid4()}",
        published_at=datetime.now(timezone.utc),
        fetched_at=datetime.now(timezone.utc),
        summary="Test summary",
        is_duplicate=False,
        duplicate_of=None,
        metadata=None,
    )


@pytest.mark.asyncio
async def test_create_article(db_connection, sample_article):
    """Test creating an article."""
    repo = SQLiteArticleRepository(db_connection)
    
    article_id = await repo.create(sample_article)
    
    assert article_id == sample_article.id


@pytest.mark.asyncio
async def test_get_article_by_id(db_connection, sample_article):
    """Test retrieving an article by ID."""
    repo = SQLiteArticleRepository(db_connection)
    
    await repo.create(sample_article)
    retrieved = await repo.get_by_id(sample_article.id)
    
    assert retrieved is not None
    assert retrieved.id == sample_article.id
    assert retrieved.title == sample_article.title
    assert retrieved.content == sample_article.content


@pytest.mark.asyncio
async def test_get_article_by_url(db_connection, sample_article):
    """Test retrieving an article by URL."""
    repo = SQLiteArticleRepository(db_connection)
    
    await repo.create(sample_article)
    retrieved = await repo.get_by_url(sample_article.url)
    
    assert retrieved is not None
    assert retrieved.url == sample_article.url


@pytest.mark.asyncio
async def test_list_articles(db_connection):
    """Test listing articles with pagination."""
    repo = SQLiteArticleRepository(db_connection)
    
    # Create multiple articles
    for i in range(5):
        article = Article(
            id=str(uuid.uuid4()),
            title=f"Article {i}",
            content=f"Content {i}",
            source="reuters",
            url=f"https://example.com/article/{i}",
            published_at=datetime.now(timezone.utc),
            fetched_at=datetime.now(timezone.utc),
        )
        await repo.create(article)
    
    # List all articles
    articles = await repo.list_articles(limit=10)
    assert len(articles) == 5
    
    # Test pagination
    articles_page1 = await repo.list_articles(skip=0, limit=2)
    assert len(articles_page1) == 2
    
    articles_page2 = await repo.list_articles(skip=2, limit=2)
    assert len(articles_page2) == 2


@pytest.mark.asyncio
async def test_count_articles(db_connection, sample_article):
    """Test counting articles."""
    repo = SQLiteArticleRepository(db_connection)
    
    initial_count = await repo.count()
    assert initial_count == 0
    
    await repo.create(sample_article)
    
    final_count = await repo.count()
    assert final_count == 1
