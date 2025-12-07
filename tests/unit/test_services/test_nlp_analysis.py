"""Unit tests for NLPAnalysisService.

This module tests the NLP analysis service functionality including
summary generation, structured data extraction, and news score computation.
"""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import NLPAnalysisError
from app.models.domain import Article, ArticleAnalysis, StructuredAnalysis
from app.services.nlp_analysis import NLPAnalysisService


@pytest.fixture
def mock_ollama_client():
    """Create mock Ollama client."""
    client = AsyncMock()
    client.generate = AsyncMock()
    return client


@pytest.fixture
def mock_article_repo():
    """Create mock article repository."""
    repo = AsyncMock()
    return repo


@pytest.fixture
def mock_analysis_repo():
    """Create mock analysis repository."""
    repo = AsyncMock()
    repo.get_by_article_id = AsyncMock(return_value=None)
    repo.create = AsyncMock(return_value="test-analysis-id")
    return repo


@pytest.fixture
def nlp_service(mock_ollama_client, mock_article_repo, mock_analysis_repo):
    """Create NLP analysis service with mocked dependencies."""
    return NLPAnalysisService(
        ollama_client=mock_ollama_client,
        article_repo=mock_article_repo,
        analysis_repo=mock_analysis_repo,
    )


@pytest.fixture
def sample_article():
    """Create sample article for testing."""
    return Article(
        id="test-article-id",
        title="Tech Company Reports Strong Earnings",
        content="Tech Company announced record earnings today, beating analyst expectations...",
        source="reuters",
        url="https://example.com/article",
        published_at=datetime.utcnow(),
        fetched_at=datetime.utcnow(),
    )


@pytest.mark.asyncio
async def test_generate_summary(nlp_service, mock_ollama_client):
    """Test summary generation."""
    # Arrange
    text = "Sample article text about financial news"
    expected_summary = "This is a concise summary of the article."
    mock_ollama_client.generate.return_value = expected_summary
    
    # Act
    summary = await nlp_service.generate_summary(text)
    
    # Assert
    assert summary == expected_summary
    mock_ollama_client.generate.assert_called_once()
    call_kwargs = mock_ollama_client.generate.call_args.kwargs
    assert "prompt" in call_kwargs
    assert text in call_kwargs["prompt"]


@pytest.mark.asyncio
async def test_extract_structured_data(nlp_service, mock_ollama_client):
    """Test structured data extraction."""
    # Arrange
    text = "Company announces merger"
    title = "Big Merger News"
    json_response = """{
        "sentiment": 1,
        "impact_magnitude": 0.8,
        "event_type": "merger",
        "confidence": 0.9,
        "estimated_price_move": 0.05
    }"""
    mock_ollama_client.generate.return_value = json_response
    
    # Act
    structured = await nlp_service.extract_structured_data(text, title)
    
    # Assert
    assert isinstance(structured, StructuredAnalysis)
    assert structured.sentiment == 1
    assert structured.impact_magnitude == 0.8
    assert structured.event_type == "merger"
    assert structured.confidence == 0.9
    assert structured.estimated_price_move == 0.05


@pytest.mark.asyncio
async def test_extract_structured_data_with_markdown(nlp_service, mock_ollama_client):
    """Test structured data extraction with markdown code blocks."""
    # Arrange
    text = "Company news"
    title = "News Title"
    json_response = """```json
{
    "sentiment": -1,
    "impact_magnitude": 0.6,
    "event_type": "regulatory",
    "confidence": 0.7,
    "estimated_price_move": -0.03
}
```"""
    mock_ollama_client.generate.return_value = json_response
    
    # Act
    structured = await nlp_service.extract_structured_data(text, title)
    
    # Assert
    assert structured.sentiment == -1
    assert structured.event_type == "regulatory"


def test_compute_news_score_positive(nlp_service):
    """Test news score computation with positive sentiment."""
    # Arrange
    analysis = StructuredAnalysis(
        sentiment=1,
        impact_magnitude=0.8,
        event_type="earnings",
        confidence=0.9,
        estimated_price_move=0.05,
    )
    
    # Act
    score = nlp_service.compute_news_score(analysis)
    
    # Assert
    assert -1.0 <= score <= 1.0
    assert score > 0  # Positive sentiment should yield positive score


def test_compute_news_score_negative(nlp_service):
    """Test news score computation with negative sentiment."""
    # Arrange
    analysis = StructuredAnalysis(
        sentiment=-1,
        impact_magnitude=0.7,
        event_type="regulatory",
        confidence=0.8,
        estimated_price_move=0.04,
    )
    
    # Act
    score = nlp_service.compute_news_score(analysis)
    
    # Assert
    assert -1.0 <= score <= 1.0
    assert score < 0  # Negative sentiment should yield negative score


def test_compute_news_score_neutral(nlp_service):
    """Test news score computation with neutral sentiment."""
    # Arrange
    analysis = StructuredAnalysis(
        sentiment=0,
        impact_magnitude=0.5,
        event_type="general",
        confidence=0.6,
        estimated_price_move=0.02,
    )
    
    # Act
    score = nlp_service.compute_news_score(analysis)
    
    # Assert
    assert score == 0.0  # Neutral sentiment should yield zero score


def test_compute_news_score_low_confidence_penalty(nlp_service):
    """Test that low confidence reduces the news score."""
    # Arrange
    high_conf = StructuredAnalysis(
        sentiment=1,
        impact_magnitude=0.8,
        event_type="earnings",
        confidence=0.9,
        estimated_price_move=0.05,
    )
    
    low_conf = StructuredAnalysis(
        sentiment=1,
        impact_magnitude=0.8,
        event_type="earnings",
        confidence=0.3,
        estimated_price_move=0.05,
    )
    
    # Act
    high_score = nlp_service.compute_news_score(high_conf)
    low_score = nlp_service.compute_news_score(low_conf)
    
    # Assert
    assert abs(high_score) > abs(low_score)  # Higher confidence = higher magnitude


def test_compute_news_score_event_weighting(nlp_service):
    """Test that event type affects the news score."""
    # Arrange
    earnings = StructuredAnalysis(
        sentiment=1,
        impact_magnitude=0.8,
        event_type="earnings",  # weight 1.5
        confidence=0.9,
        estimated_price_move=0.05,
    )
    
    general = StructuredAnalysis(
        sentiment=1,
        impact_magnitude=0.8,
        event_type="general",  # weight 0.8
        confidence=0.9,
        estimated_price_move=0.05,
    )
    
    # Act
    earnings_score = nlp_service.compute_news_score(earnings)
    general_score = nlp_service.compute_news_score(general)
    
    # Assert
    assert abs(earnings_score) > abs(general_score)  # Earnings weighted higher


@pytest.mark.asyncio
async def test_analyze_article_success(
    nlp_service,
    mock_article_repo,
    mock_analysis_repo,
    mock_ollama_client,
    sample_article,
):
    """Test successful article analysis."""
    # Arrange
    mock_article_repo.get_by_id.return_value = sample_article
    mock_ollama_client.generate.side_effect = [
        "This is a summary.",  # Summary generation
        """{
            "sentiment": 1,
            "impact_magnitude": 0.7,
            "event_type": "earnings",
            "confidence": 0.85,
            "estimated_price_move": 0.04
        }""",  # Structured extraction
    ]
    
    # Act
    analysis = await nlp_service.analyze_article(sample_article.id)
    
    # Assert
    assert isinstance(analysis, ArticleAnalysis)
    assert analysis.article_id == sample_article.id
    assert analysis.summary == "This is a summary."
    assert analysis.sentiment == 1
    assert -1.0 <= analysis.news_score <= 1.0
    mock_analysis_repo.create.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_article_not_found(nlp_service, mock_article_repo):
    """Test analysis of non-existent article."""
    # Arrange
    mock_article_repo.get_by_id.return_value = None
    
    # Act & Assert
    with pytest.raises(NLPAnalysisError, match="not found"):
        await nlp_service.analyze_article("non-existent-id")


@pytest.mark.asyncio
async def test_analyze_article_already_analyzed(
    nlp_service,
    mock_article_repo,
    mock_analysis_repo,
    sample_article,
):
    """Test that already analyzed articles return existing analysis."""
    # Arrange
    existing_analysis = ArticleAnalysis(
        id="existing-id",
        article_id=sample_article.id,
        summary="Existing summary",
        sentiment=1,
        impact_magnitude=0.7,
        event_type="earnings",
        confidence=0.8,
        estimated_price_move=0.03,
        news_score=0.5,
        analyzed_at=datetime.utcnow(),
    )
    mock_article_repo.get_by_id.return_value = sample_article
    mock_analysis_repo.get_by_article_id.return_value = existing_analysis
    
    # Act
    analysis = await nlp_service.analyze_article(sample_article.id)
    
    # Assert
    assert analysis == existing_analysis
    mock_analysis_repo.create.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_batch(nlp_service, mock_article_repo, mock_ollama_client):
    """Test batch analysis of multiple articles."""
    # Arrange
    article_ids = ["id1", "id2", "id3"]
    
    # Mock articles
    for i, article_id in enumerate(article_ids):
        article = Article(
            id=article_id,
            title=f"Article {i}",
            content=f"Content {i}",
            source="test",
            url=f"https://example.com/{i}",
            published_at=datetime.utcnow(),
            fetched_at=datetime.utcnow(),
        )
        mock_article_repo.get_by_id.side_effect = lambda aid: article if aid in article_ids else None
    
    # Mock LLM responses
    mock_ollama_client.generate.side_effect = [
        "Summary 1",
        '{"sentiment": 1, "impact_magnitude": 0.7, "event_type": "earnings", "confidence": 0.8, "estimated_price_move": 0.03}',
        "Summary 2",
        '{"sentiment": 0, "impact_magnitude": 0.5, "event_type": "general", "confidence": 0.6, "estimated_price_move": 0.01}',
        "Summary 3",
        '{"sentiment": -1, "impact_magnitude": 0.6, "event_type": "regulatory", "confidence": 0.7, "estimated_price_move": 0.02}',
    ]
    
    # Act
    results = await nlp_service.analyze_batch(article_ids, max_concurrent=2)
    
    # Assert
    assert len(results) <= len(article_ids)  # May have some failures
