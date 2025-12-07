"""Service layer for business logic.

This module exports all service classes for easy importing.
"""

from app.services.news_ingestion import NewsIngestionService
from app.services.nlp_analysis import NLPAnalysisService

__all__ = [
    "NewsIngestionService",
    "NLPAnalysisService",
]
