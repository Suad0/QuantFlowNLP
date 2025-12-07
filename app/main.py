"""Main FastAPI application entry point.

This module initializes the FastAPI application with all routers, middleware,
and lifecycle events for the Quantitative Trading Intelligence System.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.core.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events.
    
    This context manager handles startup and shutdown events for the application,
    including database initialization, connection setup, and cleanup.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("=" * 80)
    logger.info("Starting Quantitative Trading Intelligence System")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Database: {settings.database_url}")
    logger.info(f"Ollama URL: {settings.ollama_base_url}")
    logger.info(f"ChromaDB: {settings.chromadb_host}:{settings.chromadb_port}")
    logger.info("=" * 80)
    
    # Initialize database connection pool
    try:
        from app.adapters.database import init_database
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    # Initialize ChromaDB client
    try:
        from app.adapters.chromadb_client import ChromaDBClient
        chroma_client = ChromaDBClient()
        is_healthy = await chroma_client.health_check()
        if is_healthy:
            logger.info("ChromaDB client initialized successfully")
        else:
            logger.warning("ChromaDB client initialized but health check failed")
    except Exception as e:
        logger.warning(f"Failed to initialize ChromaDB: {e}")
        logger.warning("ChromaDB features will be unavailable")
    
    # Verify Ollama connection
    try:
        from app.adapters.ollama_client import OllamaClient
        ollama_client = OllamaClient()
        is_healthy = await ollama_client.health_check()
        if is_healthy:
            logger.info("Ollama service is healthy")
        else:
            logger.warning("Ollama service health check failed - some features may not work")
    except Exception as e:
        logger.warning(f"Could not verify Ollama connection: {e}")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Quantitative Trading Intelligence System")
    
    # Close database connections
    try:
        from app.adapters.database import close_database
        await close_database()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")
    
    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=(
        "Local Quantitative Trading Intelligence System - A production-ready Python "
        "application that ingests financial news, performs local NLP analysis using Ollama, "
        "prepares features for xLSTM forecasting models, generates predictions, and provides "
        "portfolio optimization capabilities."
    ),
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip compression middleware
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,  # Only compress responses larger than 1KB
)


# Import and include routers
from app.api.routers import features, health, news, nlp, portfolio, predictions, quant

app.include_router(news.router, prefix="/api/v1", tags=["news"])
app.include_router(nlp.router, prefix="/api/v1", tags=["nlp"])
app.include_router(features.router, prefix="/api/v1", tags=["features"])
app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
app.include_router(portfolio.router, prefix="/api/v1", tags=["portfolio"])
app.include_router(quant.router, prefix="/api/v1", tags=["quant"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])


@app.get("/")
async def root():
    """Root endpoint providing API information.
    
    Returns:
        Basic API information and links to documentation
    """
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "status": "operational",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
