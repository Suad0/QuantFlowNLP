"""Dependency injection for FastAPI routes.

This module provides dependency injection functions for all services and repositories,
enabling clean separation of concerns and easy testing.
"""

from typing import Annotated

from fastapi import Depends

from app.adapters.chromadb_client import ChromaDBClient
from app.repositories.vector_repository import ChromaVectorRepository
from app.adapters.database import get_database
from app.adapters.news_sources.base import NewsSourceAdapter
from app.adapters.news_sources.bloomberg import BloombergAdapter
from app.adapters.news_sources.investing_com import InvestingComAdapter
from app.adapters.news_sources.marketwatch import MarketWatchAdapter
from app.adapters.news_sources.reuters import ReutersAdapter
from app.adapters.news_sources.yahoo_finance import YahooFinanceAdapter
from app.adapters.ollama_client import OllamaClient
from app.repositories.analysis_repository import SQLiteAnalysisRepository
from app.repositories.article_repository import SQLiteArticleRepository
from app.repositories.ohlcv_repository import SQLiteOHLCVRepository
from app.repositories.scaler_repository import SQLiteScalerRepository
from app.services.feature_engineering import FeatureEngineeringService
from app.services.news_ingestion import NewsIngestionService
from app.services.nlp_analysis import NLPAnalysisService
from app.services.portfolio_optimization import PortfolioOptimizationService
from app.services.prediction import PredictionService
from app.services.quantitative_finance import QuantitativeFinanceService


# ============================================================================
# Repository Dependencies
# ============================================================================


async def get_article_repository() -> SQLiteArticleRepository:
    """Get article repository instance.
    
    Returns:
        Configured article repository
    """
    db = await get_database()
    return SQLiteArticleRepository(db)


async def get_analysis_repository() -> SQLiteAnalysisRepository:
    """Get analysis repository instance.
    
    Returns:
        Configured analysis repository
    """
    db = await get_database()
    return SQLiteAnalysisRepository(db)


async def get_ohlcv_repository() -> SQLiteOHLCVRepository:
    """Get OHLCV repository instance.
    
    Returns:
        Configured OHLCV repository
    """
    db = await get_database()
    return SQLiteOHLCVRepository(db)


async def get_scaler_repository() -> SQLiteScalerRepository:
    """Get scaler repository instance.
    
    Returns:
        Configured scaler repository
    """
    db = await get_database()
    return SQLiteScalerRepository(db)


async def get_vector_repository() -> ChromaVectorRepository:
    """Get vector repository instance.
    
    Returns:
        Configured vector repository
    """
    from app.core.config import settings
    
    client = ChromaDBClient(use_persistent_client=settings.chromadb_use_persistent)
    await client.connect()
    await client.get_or_create_collection()
    return ChromaVectorRepository(client)


# ============================================================================
# Adapter Dependencies
# ============================================================================


def get_ollama_client() -> OllamaClient:
    """Get Ollama client instance.
    
    Returns:
        Configured Ollama client
    """
    return OllamaClient()


def get_news_sources() -> list[NewsSourceAdapter]:
    """Get list of configured news source adapters.
    
    Returns:
        List of news source adapters
    """
    return [
        YahooFinanceAdapter(),
        ReutersAdapter(),
        BloombergAdapter(),
        MarketWatchAdapter(),
        InvestingComAdapter(),
    ]


# ============================================================================
# Service Dependencies
# ============================================================================


async def get_news_ingestion_service(
    article_repo: Annotated[SQLiteArticleRepository, Depends(get_article_repository)],
    vector_repo: Annotated[ChromaVectorRepository, Depends(get_vector_repository)],
    ollama_client: Annotated[OllamaClient, Depends(get_ollama_client)],
    news_sources: Annotated[list[NewsSourceAdapter], Depends(get_news_sources)],
) -> NewsIngestionService:
    """Get news ingestion service instance.
    
    Args:
        article_repo: Article repository
        vector_repo: Vector repository
        ollama_client: Ollama client
        news_sources: List of news sources
    
    Returns:
        Configured news ingestion service
    """
    return NewsIngestionService(
        article_repo=article_repo,
        vector_repo=vector_repo,
        ollama_client=ollama_client,
        news_sources=news_sources,
    )


async def get_nlp_service(
    ollama_client: Annotated[OllamaClient, Depends(get_ollama_client)],
    article_repo: Annotated[SQLiteArticleRepository, Depends(get_article_repository)],
    analysis_repo: Annotated[SQLiteAnalysisRepository, Depends(get_analysis_repository)],
) -> NLPAnalysisService:
    """Get NLP analysis service instance.
    
    Args:
        ollama_client: Ollama client
        article_repo: Article repository
        analysis_repo: Analysis repository
    
    Returns:
        Configured NLP analysis service
    """
    return NLPAnalysisService(
        ollama_client=ollama_client,
        article_repo=article_repo,
        analysis_repo=analysis_repo,
    )


async def get_feature_service(
    ohlcv_repo: Annotated[SQLiteOHLCVRepository, Depends(get_ohlcv_repository)],
    analysis_repo: Annotated[SQLiteAnalysisRepository, Depends(get_analysis_repository)],
    scaler_repo: Annotated[SQLiteScalerRepository, Depends(get_scaler_repository)],
) -> FeatureEngineeringService:
    """Get feature engineering service instance.
    
    Args:
        ohlcv_repo: OHLCV repository
        analysis_repo: Analysis repository
        scaler_repo: Scaler repository
    
    Returns:
        Configured feature engineering service
    """
    return FeatureEngineeringService(
        ohlcv_repository=ohlcv_repo,
        analysis_repository=analysis_repo,
        scaler_repository=scaler_repo,
    )


def get_prediction_service() -> PredictionService:
    """Get prediction service instance.
    
    Returns:
        Configured prediction service
    """
    return PredictionService()


async def get_portfolio_service(
    ohlcv_repo: Annotated[SQLiteOHLCVRepository, Depends(get_ohlcv_repository)],
) -> PortfolioOptimizationService:
    """Get portfolio optimization service instance.
    
    Args:
        ohlcv_repo: OHLCV repository
    
    Returns:
        Configured portfolio optimization service
    """
    return PortfolioOptimizationService(ohlcv_repository=ohlcv_repo)


def get_quant_service() -> QuantitativeFinanceService:
    """Get quantitative finance service instance.
    
    Returns:
        Configured quantitative finance service
    """
    return QuantitativeFinanceService()


# ============================================================================
# Type Aliases for Dependency Injection
# ============================================================================

# These type aliases make it easier to use dependencies in route handlers
ArticleRepositoryDep = Annotated[SQLiteArticleRepository, Depends(get_article_repository)]
AnalysisRepositoryDep = Annotated[SQLiteAnalysisRepository, Depends(get_analysis_repository)]
OHLCVRepositoryDep = Annotated[SQLiteOHLCVRepository, Depends(get_ohlcv_repository)]
ScalerRepositoryDep = Annotated[SQLiteScalerRepository, Depends(get_scaler_repository)]
VectorRepositoryDep = Annotated[ChromaVectorRepository, Depends(get_vector_repository)]

OllamaClientDep = Annotated[OllamaClient, Depends(get_ollama_client)]
NewsSourcesDep = Annotated[list[NewsSourceAdapter], Depends(get_news_sources)]

NewsIngestionServiceDep = Annotated[NewsIngestionService, Depends(get_news_ingestion_service)]
NLPServiceDep = Annotated[NLPAnalysisService, Depends(get_nlp_service)]
FeatureServiceDep = Annotated[FeatureEngineeringService, Depends(get_feature_service)]
PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]
PortfolioServiceDep = Annotated[PortfolioOptimizationService, Depends(get_portfolio_service)]
QuantServiceDep = Annotated[QuantitativeFinanceService, Depends(get_quant_service)]
