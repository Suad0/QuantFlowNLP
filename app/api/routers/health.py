"""Health check and monitoring API router.

This module provides REST endpoints for checking the health status
of the application and its dependencies.
"""

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, status

from app.adapters.chromadb_client import ChromaDBClient
from app.adapters.database import get_database
from app.adapters.ollama_client import OllamaClient
from app.core.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Overall health check",
    description="Check the overall health status of the application and all dependencies.",
)
async def health_check() -> Dict[str, Any]:
    """Check overall application health.
    
    Performs health checks on all critical components:
    - Database connectivity
    - ChromaDB connectivity
    - Ollama service availability
    
    Returns:
        Health status for all components
    """
    logger.debug("Performing overall health check")
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.api_version,
        "environment": settings.environment,
        "components": {},
    }
    
    # Check database
    try:
        db = await get_database()
        await db.execute("SELECT 1")
        health_status["components"]["database"] = {
            "status": "healthy",
            "message": "Database connection successful",
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}",
        }
        health_status["status"] = "degraded"
    
    # Check ChromaDB
    try:
        chroma_client = ChromaDBClient()
        is_healthy = await chroma_client.health_check()
        if is_healthy:
            health_status["components"]["chromadb"] = {
                "status": "healthy",
                "message": "ChromaDB connection successful",
            }
        else:
            health_status["components"]["chromadb"] = {
                "status": "unhealthy",
                "message": "ChromaDB health check failed",
            }
            health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"ChromaDB health check failed: {e}")
        health_status["components"]["chromadb"] = {
            "status": "unhealthy",
            "message": f"ChromaDB connection failed: {str(e)}",
        }
        health_status["status"] = "degraded"
    
    # Check Ollama
    try:
        ollama_client = OllamaClient()
        is_healthy = await ollama_client.health_check()
        if is_healthy:
            health_status["components"]["ollama"] = {
                "status": "healthy",
                "message": "Ollama service is available",
            }
        else:
            health_status["components"]["ollama"] = {
                "status": "unhealthy",
                "message": "Ollama service health check failed",
            }
            health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        health_status["components"]["ollama"] = {
            "status": "unhealthy",
            "message": f"Ollama service unavailable: {str(e)}",
        }
        health_status["status"] = "degraded"
    
    logger.info(
        f"Health check completed: {health_status['status']}",
        extra={"components": health_status["components"]},
    )
    
    return health_status


@router.get(
    "/health/database",
    status_code=status.HTTP_200_OK,
    summary="Database health check",
    description="Check database connectivity and basic operations.",
)
async def database_health_check() -> Dict[str, Any]:
    """Check database health.
    
    Verifies that the database is accessible and can execute queries.
    
    Returns:
        Database health status
    """
    logger.debug("Performing database health check")
    
    try:
        db = await get_database()
        
        # Test basic query
        cursor = await db.execute("SELECT 1")
        result = await cursor.fetchone()
        
        # Get database info
        cursor = await db.execute("PRAGMA database_list")
        db_info = await cursor.fetchall()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Database is accessible",
            "details": {
                "database_url": settings.database_url,
                "query_test": "passed",
                "database_count": len(db_info),
            },
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"Database health check failed: {str(e)}",
            "details": {
                "database_url": settings.database_url,
                "error": str(e),
            },
        }


@router.get(
    "/health/chromadb",
    status_code=status.HTTP_200_OK,
    summary="ChromaDB health check",
    description="Check ChromaDB connectivity and collection status.",
)
async def chromadb_health_check() -> Dict[str, Any]:
    """Check ChromaDB health.
    
    Verifies that ChromaDB is accessible and can list collections.
    
    Returns:
        ChromaDB health status
    """
    logger.debug("Performing ChromaDB health check")
    
    try:
        chroma_client = ChromaDBClient()
        is_healthy = await chroma_client.health_check()
        
        if is_healthy:
            # Try to get collection info
            await chroma_client.connect()
            collection = await chroma_client.get_or_create_collection()
            count = await chroma_client.count()
            
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "ChromaDB is accessible",
                "details": {
                    "host": settings.chromadb_host,
                    "port": settings.chromadb_port,
                    "collection_name": chroma_client.collection_name,
                    "embedding_count": count,
                },
            }
        else:
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "ChromaDB health check failed",
                "details": {
                    "host": settings.chromadb_host,
                    "port": settings.chromadb_port,
                },
            }
        
    except Exception as e:
        logger.error(f"ChromaDB health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"ChromaDB health check failed: {str(e)}",
            "details": {
                "host": settings.chromadb_host,
                "port": settings.chromadb_port,
                "error": str(e),
            },
        }


@router.get(
    "/health/ollama",
    status_code=status.HTTP_200_OK,
    summary="Ollama service health check",
    description="Check Ollama service availability and model status.",
)
async def ollama_health_check() -> Dict[str, Any]:
    """Check Ollama service health.
    
    Verifies that Ollama is accessible and can respond to requests.
    
    Returns:
        Ollama health status
    """
    logger.debug("Performing Ollama health check")
    
    try:
        ollama_client = OllamaClient()
        
        # Perform health check
        is_healthy = await ollama_client.health_check()
        
        if is_healthy:
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Ollama service is available",
                "details": {
                    "base_url": settings.ollama_base_url,
                    "llm_model": settings.ollama_llm_model,
                    "embedding_model": settings.ollama_embedding_model,
                    "health_check": "passed",
                },
            }
        else:
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Ollama service health check failed",
                "details": {
                    "base_url": settings.ollama_base_url,
                    "health_check": "failed",
                },
            }
            
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"Ollama service unavailable: {str(e)}",
            "details": {
                "base_url": settings.ollama_base_url,
                "error": str(e),
            },
        }
