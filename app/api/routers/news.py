"""News ingestion API router.

This module provides REST endpoints for news ingestion and article retrieval.
"""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.api.dependencies import (
    ArticleRepositoryDep,
    NewsIngestionServiceDep,
)
from app.core.exceptions import NewsIngestionError
from app.models.api.news import (
    ArticleListResponse,
    ArticleResponse,
    IngestNewsRequest,
    IngestionResult,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/ingest-news",
    response_model=IngestionResult,
    status_code=status.HTTP_200_OK,
    summary="Ingest news from multiple sources",
    description=(
        "Fetch financial news articles from configured sources, generate embeddings, "
        "check for duplicates, and store in the database. Returns statistics about "
        "the ingestion process including per-source breakdowns."
    ),
)
async def ingest_news(
    request: IngestNewsRequest,
    news_service: NewsIngestionServiceDep,
) -> IngestionResult:
    """Ingest news articles from multiple sources.
    
    This endpoint orchestrates the complete news ingestion pipeline:
    1. Fetch articles from specified sources (or all if not specified)
    2. Generate embeddings for each article
    3. Check for duplicates using similarity matching
    4. Store new articles in database and ChromaDB
    
    Args:
        request: Ingestion request with optional source filters
        news_service: News ingestion service (injected)
    
    Returns:
        Ingestion statistics including counts and errors
    
    Raises:
        HTTPException: If ingestion fails
    """
    logger.info(
        "Received news ingestion request",
        extra={
            "sources": request.sources,
            "max_articles_per_source": request.max_articles_per_source,
        },
    )
    
    start_time = time.time()
    
    try:
        # Perform ingestion
        result = await news_service.ingest_from_all_sources(
            max_articles_per_source=request.max_articles_per_source,
        )
        
        duration = time.time() - start_time
        
        logger.info(
            "News ingestion completed",
            extra={
                "total_fetched": result.get("total_fetched", 0),
                "total_stored": result.get("total_stored", 0),
                "duplicates_found": result.get("duplicates_found", 0),
                "duration_seconds": duration,
            },
        )
        
        # Convert to response model
        return IngestionResult(
            total_fetched=result.get("total_fetched", 0),
            total_stored=result.get("total_stored", 0),
            duplicates_found=result.get("duplicates_found", 0),
            errors=result.get("errors", []),
            duration_seconds=duration,
            by_source=result.get("by_source", {}),
        )
        
    except NewsIngestionError as e:
        logger.error(f"News ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"News ingestion failed: {str(e)}",
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error during news ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        ) from e


@router.get(
    "/articles",
    response_model=ArticleListResponse,
    status_code=status.HTTP_200_OK,
    summary="List articles with pagination",
    description=(
        "Retrieve a paginated list of articles from the database. "
        "Supports filtering and pagination parameters."
    ),
)
async def list_articles(
    article_repo: ArticleRepositoryDep,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of records to return"),
    source: Optional[str] = Query(None, description="Filter by news source"),
) -> ArticleListResponse:
    """List articles with pagination.
    
    Args:
        article_repo: Article repository (injected)
        skip: Number of records to skip (pagination offset)
        limit: Maximum number of records to return
        source: Optional source filter
    
    Returns:
        Paginated list of articles
    
    Raises:
        HTTPException: If retrieval fails
    """
    logger.debug(
        f"Listing articles: skip={skip}, limit={limit}, source={source}"
    )
    
    try:
        # Get articles from repository
        articles = await article_repo.list_articles(skip=skip, limit=limit)
        
        # Get total count (for pagination)
        # Note: This is a simplified version. In production, you'd want to
        # implement a count method in the repository
        total = len(articles) + skip  # Approximate
        
        # Convert domain models to API response models
        article_responses = [
            ArticleResponse(
                id=article.id,
                title=article.title,
                content=article.content,
                summary=article.summary,
                source=article.source,
                url=article.url,
                published_at=article.published_at,
                fetched_at=article.fetched_at,
                is_duplicate=article.is_duplicate,
                duplicate_of=article.duplicate_of,
            )
            for article in articles
        ]
        
        logger.debug(f"Retrieved {len(article_responses)} articles")
        
        return ArticleListResponse(
            articles=article_responses,
            total=total,
            skip=skip,
            limit=limit,
        )
        
    except Exception as e:
        logger.error(f"Failed to list articles: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve articles: {str(e)}",
        ) from e


@router.get(
    "/article/{article_id}",
    response_model=ArticleResponse,
    status_code=status.HTTP_200_OK,
    summary="Get article by ID",
    description="Retrieve a single article by its unique identifier.",
)
async def get_article(
    article_id: str,
    article_repo: ArticleRepositoryDep,
) -> ArticleResponse:
    """Get a single article by ID.
    
    Args:
        article_id: Article unique identifier
        article_repo: Article repository (injected)
    
    Returns:
        Article details
    
    Raises:
        HTTPException: If article not found or retrieval fails
    """
    logger.debug(f"Retrieving article: {article_id}")
    
    try:
        # Get article from repository
        article = await article_repo.get_by_id(article_id)
        
        if not article:
            logger.warning(f"Article not found: {article_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Article with ID {article_id} not found",
            )
        
        logger.debug(f"Retrieved article: {article_id}")
        
        # Convert to API response model
        return ArticleResponse(
            id=article.id,
            title=article.title,
            content=article.content,
            summary=article.summary,
            source=article.source,
            url=article.url,
            published_at=article.published_at,
            fetched_at=article.fetched_at,
            is_duplicate=article.is_duplicate,
            duplicate_of=article.duplicate_of,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve article {article_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve article: {str(e)}",
        ) from e
