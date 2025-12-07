"""NLP analysis API router.

This module provides REST endpoints for article NLP analysis including
sentiment extraction, impact assessment, and news score computation.
"""

import uuid
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.api.dependencies import NLPServiceDep
from app.core.exceptions import NLPAnalysisError
from app.models.api.nlp import (
    AnalyzeBatchRequest,
    AnalyzeBatchResponse,
    AnalyzeArticleResponse,
    TaskStatusResponse,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# In-memory task storage (in production, use Redis or database)
_task_storage: Dict[str, Dict] = {}


async def _analyze_batch_background(
    task_id: str,
    article_ids: list[str],
    nlp_service: NLPServiceDep,
) -> None:
    """Background task for batch analysis.
    
    Args:
        task_id: Task identifier
        article_ids: List of article IDs to analyze
        nlp_service: NLP service instance
    """
    try:
        _task_storage[task_id]["status"] = "processing"
        _task_storage[task_id]["progress"] = {
            "total": len(article_ids),
            "completed": 0,
            "failed": 0,
        }
        
        results = []
        for i, article_id in enumerate(article_ids):
            try:
                analysis = await nlp_service.analyze_article(article_id)
                results.append({
                    "article_id": article_id,
                    "success": True,
                    "news_score": analysis.news_score,
                })
                _task_storage[task_id]["progress"]["completed"] += 1
            except Exception as e:
                logger.error(f"Failed to analyze article {article_id}: {e}")
                results.append({
                    "article_id": article_id,
                    "success": False,
                    "error": str(e),
                })
                _task_storage[task_id]["progress"]["failed"] += 1
        
        _task_storage[task_id]["status"] = "completed"
        _task_storage[task_id]["results"] = {
            "analyses": results,
            "summary": {
                "total": len(article_ids),
                "successful": _task_storage[task_id]["progress"]["completed"],
                "failed": _task_storage[task_id]["progress"]["failed"],
            },
        }
        
        logger.info(f"Batch analysis task {task_id} completed")
        
    except Exception as e:
        logger.error(f"Batch analysis task {task_id} failed: {e}", exc_info=True)
        _task_storage[task_id]["status"] = "failed"
        _task_storage[task_id]["error"] = str(e)


@router.post(
    "/analyze-article/{article_id}",
    response_model=AnalyzeArticleResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze a single article",
    description=(
        "Perform NLP analysis on a single article using local LLM. "
        "Extracts sentiment, impact, event type, and computes news score."
    ),
)
async def analyze_article(
    article_id: str,
    nlp_service: NLPServiceDep,
) -> AnalyzeArticleResponse:
    """Analyze a single article.
    
    Performs complete NLP analysis including:
    - Summary generation
    - Sentiment extraction
    - Impact magnitude assessment
    - Event type classification
    - News score computation
    
    Args:
        article_id: Article unique identifier
        nlp_service: NLP service (injected)
    
    Returns:
        Complete article analysis
    
    Raises:
        HTTPException: If analysis fails or article not found
    """
    logger.info(f"Analyzing article: {article_id}")
    
    try:
        # Perform analysis
        analysis = await nlp_service.analyze_article(article_id)
        
        logger.info(
            f"Article analysis completed: {article_id}",
            extra={
                "news_score": analysis.news_score,
                "sentiment": analysis.structured.sentiment,
                "confidence": analysis.structured.confidence,
            },
        )
        
        # Convert to API response model
        return AnalyzeArticleResponse(
            article_id=analysis.article_id,
            summary=analysis.summary,
            sentiment=analysis.structured.sentiment,
            impact_magnitude=analysis.structured.impact_magnitude,
            event_type=analysis.structured.event_type,
            confidence=analysis.structured.confidence,
            estimated_price_move=analysis.structured.estimated_price_move,
            news_score=analysis.news_score,
            analyzed_at=analysis.analyzed_at,
        )
        
    except NLPAnalysisError as e:
        logger.error(f"NLP analysis failed for article {article_id}: {e}")
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Article {article_id} not found",
            ) from e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        ) from e
    except Exception as e:
        logger.error(
            f"Unexpected error analyzing article {article_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        ) from e


@router.post(
    "/analyze-latest",
    response_model=AnalyzeBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Analyze latest unanalyzed articles",
    description=(
        "Queue batch analysis of latest unanalyzed articles. "
        "Processing happens in background. Returns task ID for status checking."
    ),
)
async def analyze_latest(
    request: AnalyzeBatchRequest,
    background_tasks: BackgroundTasks,
    nlp_service: NLPServiceDep,
) -> AnalyzeBatchResponse:
    """Analyze latest unanalyzed articles in batch.
    
    This endpoint queues a background task to analyze multiple articles.
    Use the returned task_id to check progress via the status endpoint.
    
    Args:
        request: Batch analysis request
        background_tasks: FastAPI background tasks
        nlp_service: NLP service (injected)
    
    Returns:
        Task information including task_id
    
    Raises:
        HTTPException: If task creation fails
    """
    logger.info(
        "Received batch analysis request",
        extra={"limit": request.limit, "article_ids": request.article_ids},
    )
    
    try:
        # Get articles to analyze
        if request.article_ids:
            article_ids = request.article_ids
        else:
            # Get unanalyzed articles from service
            # Note: This requires implementing get_unanalyzed in the service
            # For now, we'll use a placeholder
            article_ids = []
            logger.warning("Getting unanalyzed articles not yet implemented")
        
        if not article_ids:
            return AnalyzeBatchResponse(
                task_id="",
                article_count=0,
                status="completed",
                message="No articles to analyze",
            )
        
        # Apply limit if specified
        if request.limit:
            article_ids = article_ids[:request.limit]
        
        # Create task
        task_id = str(uuid.uuid4())
        _task_storage[task_id] = {
            "status": "queued",
            "article_count": len(article_ids),
        }
        
        # Queue background task
        background_tasks.add_task(
            _analyze_batch_background,
            task_id,
            article_ids,
            nlp_service,
        )
        
        logger.info(
            f"Batch analysis task created: {task_id}",
            extra={"article_count": len(article_ids)},
        )
        
        return AnalyzeBatchResponse(
            task_id=task_id,
            article_count=len(article_ids),
            status="processing",
            message=f"Analyzing {len(article_ids)} articles",
        )
        
    except Exception as e:
        logger.error(f"Failed to create batch analysis task: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create analysis task: {str(e)}",
        ) from e


@router.get(
    "/analysis/status/{task_id}",
    response_model=TaskStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Check batch analysis task status",
    description="Check the status and progress of a batch analysis task.",
)
async def get_task_status(
    task_id: str,
) -> TaskStatusResponse:
    """Get status of a batch analysis task.
    
    Args:
        task_id: Task identifier
    
    Returns:
        Task status and progress information
    
    Raises:
        HTTPException: If task not found
    """
    logger.debug(f"Checking status for task: {task_id}")
    
    if task_id not in _task_storage:
        logger.warning(f"Task not found: {task_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )
    
    task_data = _task_storage[task_id]
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_data["status"],
        progress=task_data.get("progress"),
        results=task_data.get("results"),
        error=task_data.get("error"),
    )
