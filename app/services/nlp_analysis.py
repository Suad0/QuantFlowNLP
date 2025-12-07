"""NLP Analysis Service for article sentiment and impact analysis.

This module provides the NLPAnalysisService which performs local LLM-based
analysis of financial news articles using Ollama, extracting structured insights
and computing news scores.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import List, Optional

import numpy as np

from app.adapters.ollama_client import OllamaClient
from app.core.config import settings
from app.core.exceptions import NLPAnalysisError
from app.models.domain import Article, ArticleAnalysis, StructuredAnalysis
from app.repositories.base import AnalysisRepository, ArticleRepository

logger = logging.getLogger(__name__)


class NLPAnalysisService:
    """Service for performing NLP analysis on financial news articles.
    
    Uses Ollama's local LLM (llama3) for text generation and analysis,
    extracting sentiment, impact, and computing aggregated news scores.
    
    Attributes:
        ollama_client: Client for Ollama API
        article_repo: Repository for article operations
        analysis_repo: Repository for analysis operations
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        article_repo: ArticleRepository,
        analysis_repo: AnalysisRepository,
    ):
        """Initialize NLP analysis service.
        
        Args:
            ollama_client: Ollama client for LLM inference
            article_repo: Article repository
            analysis_repo: Analysis repository
        """
        self.ollama_client = ollama_client
        self.article_repo = article_repo
        self.analysis_repo = analysis_repo
        
        logger.info("Initialized NLPAnalysisService")

    async def analyze_article(self, article_id: str) -> ArticleAnalysis:
        """Analyze a single article and store the results.
        
        Performs complete NLP analysis including summary generation,
        structured data extraction, and news score computation.
        
        Args:
            article_id: ID of article to analyze
            
        Returns:
            Complete article analysis
            
        Raises:
            NLPAnalysisError: If analysis fails
        """
        logger.info(f"Starting analysis for article: {article_id}")
        
        try:
            # Retrieve article
            article = await self.article_repo.get_by_id(article_id)
            if not article:
                raise NLPAnalysisError(f"Article {article_id} not found")
            
            # Check if already analyzed
            existing_analysis = await self.analysis_repo.get_by_article_id(article_id)
            if existing_analysis:
                logger.info(f"Article {article_id} already analyzed, returning existing analysis")
                return existing_analysis
            
            # Generate summary
            logger.debug(f"Generating summary for article: {article_id}")
            summary = await self.generate_summary(article.content)
            
            # Extract structured data
            logger.debug(f"Extracting structured data for article: {article_id}")
            structured = await self.extract_structured_data(article.content, article.title)
            
            # Compute news score
            logger.debug(f"Computing news score for article: {article_id}")
            news_score = self.compute_news_score(structured)
            
            # Create analysis object
            analysis = ArticleAnalysis(
                id=str(uuid.uuid4()),
                article_id=article_id,
                summary=summary,
                sentiment=structured.sentiment,
                impact_magnitude=structured.impact_magnitude,
                event_type=structured.event_type,
                confidence=structured.confidence,
                estimated_price_move=structured.estimated_price_move,
                news_score=news_score,
                analyzed_at=datetime.utcnow(),
                metadata=None,
            )
            
            # Store analysis
            await self.analysis_repo.create(analysis)
            
            logger.info(
                f"Completed analysis for article {article_id}: "
                f"sentiment={structured.sentiment}, score={news_score:.3f}"
            )
            
            return analysis
            
        except NLPAnalysisError:
            raise
        except Exception as e:
            logger.error(f"Failed to analyze article {article_id}: {e}")
            raise NLPAnalysisError(f"Analysis failed for article {article_id}: {e}") from e

    async def generate_summary(self, text: str) -> str:
        """Generate a concise summary of article text using Ollama.
        
        Uses prompt engineering to ensure consistent, high-quality summaries
        focused on financial impact and key events.
        
        Args:
            text: Article text to summarize
            
        Returns:
            Generated summary (max 200 tokens)
            
        Raises:
            NLPAnalysisError: If summary generation fails
        """
        try:
            # Construct prompt for summary generation
            prompt = self._build_summary_prompt(text)
            
            # Generate summary using Ollama
            summary = await self.ollama_client.generate(
                model=settings.ollama_llm_model,
                prompt=prompt,
                temperature=settings.nlp_analysis_temperature,
                max_tokens=settings.nlp_summary_max_tokens,
            )
            
            # Clean up summary
            summary = summary.strip()
            
            if not summary:
                raise NLPAnalysisError("Generated summary is empty")
            
            logger.debug(f"Generated summary of length: {len(summary)}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            raise NLPAnalysisError(f"Summary generation failed: {e}") from e

    def _build_summary_prompt(self, text: str) -> str:
        """Build prompt for summary generation.
        
        Args:
            text: Article text
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a financial analyst summarizing news articles for traders.

Summarize the following financial news article in 2-3 concise sentences. Focus on:
- Key events and their financial implications
- Market impact and affected companies/sectors
- Actionable insights for traders

Article:
{text}

Summary:"""
        
        return prompt

    async def extract_structured_data(self, text: str, title: str) -> StructuredAnalysis:
        """Extract structured data from article using Ollama.
        
        Extracts sentiment, impact magnitude, event type, confidence,
        and estimated price move using prompt engineering and JSON parsing.
        
        Args:
            text: Article text
            title: Article title
            
        Returns:
            Structured analysis data
            
        Raises:
            NLPAnalysisError: If extraction fails
        """
        try:
            # Construct prompt for structured extraction
            prompt = self._build_extraction_prompt(text, title)
            
            # Generate structured data using Ollama
            response = await self.ollama_client.generate(
                model=settings.ollama_llm_model,
                prompt=prompt,
                temperature=settings.nlp_analysis_temperature,
            )
            
            # Parse JSON response
            structured_data = self._parse_extraction_response(response)
            
            # Validate extracted fields
            validated = self._validate_structured_data(structured_data)
            
            logger.debug(
                f"Extracted structured data: sentiment={validated.sentiment}, "
                f"impact={validated.impact_magnitude:.2f}, "
                f"event_type={validated.event_type}"
            )
            
            return validated
            
        except Exception as e:
            logger.error(f"Failed to extract structured data: {e}")
            raise NLPAnalysisError(f"Structured data extraction failed: {e}") from e

    def _build_extraction_prompt(self, text: str, title: str) -> str:
        """Build prompt for structured data extraction.
        
        Args:
            text: Article text
            title: Article title
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a financial analyst extracting structured data from news articles.

Analyze the following financial news article and extract structured information.

Title: {title}

Article:
{text}

Extract the following information and respond ONLY with valid JSON (no additional text):

{{
  "sentiment": <integer: -1 for negative, 0 for neutral, 1 for positive>,
  "impact_magnitude": <float: 0.0 to 1.0, how significant is this news>,
  "event_type": <string: one of "earnings", "merger", "regulatory", "product_launch", "general">,
  "confidence": <float: 0.0 to 1.0, your confidence in this analysis>,
  "estimated_price_move": <float: estimated percentage price move, e.g., 0.05 for 5%>
}}

Guidelines:
- sentiment: Consider overall market impact (positive/negative/neutral)
- impact_magnitude: 0.0 = negligible, 0.5 = moderate, 1.0 = major market-moving event
- event_type: Choose the most relevant category
- confidence: How certain are you about this analysis
- estimated_price_move: Estimated percentage change (can be negative)

JSON response:"""
        
        return prompt

    def _parse_extraction_response(self, response: str) -> dict:  # type: ignore[type-arg]
        """Parse JSON response from LLM.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed dictionary
            
        Raises:
            NLPAnalysisError: If parsing fails
        """
        try:
            # Try to find JSON in response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```"):
                lines = response.split("\n")
                # Remove first and last lines (``` markers)
                response = "\n".join(lines[1:-1])
                if response.startswith("json"):
                    response = "\n".join(response.split("\n")[1:])
            
            # Parse JSON
            data = json.loads(response)
            
            # Validate required fields
            required_fields = [
                "sentiment",
                "impact_magnitude",
                "event_type",
                "confidence",
                "estimated_price_move",
            ]
            
            for field in required_fields:
                if field not in data:
                    raise NLPAnalysisError(f"Missing required field: {field}")
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response}")
            raise NLPAnalysisError(f"Invalid JSON response: {e}") from e
        except Exception as e:
            logger.error(f"Failed to parse extraction response: {e}")
            raise NLPAnalysisError(f"Response parsing failed: {e}") from e

    def _validate_structured_data(self, data: dict) -> StructuredAnalysis:  # type: ignore[type-arg]
        """Validate and convert extracted data to StructuredAnalysis.
        
        Args:
            data: Extracted data dictionary
            
        Returns:
            Validated StructuredAnalysis object
            
        Raises:
            NLPAnalysisError: If validation fails
        """
        try:
            # Validate and convert sentiment
            sentiment = int(data["sentiment"])
            if sentiment not in [-1, 0, 1]:
                logger.warning(f"Invalid sentiment {sentiment}, clamping to valid range")
                sentiment = max(-1, min(1, sentiment))
            
            # Validate and convert impact_magnitude
            impact_magnitude = float(data["impact_magnitude"])
            if not 0.0 <= impact_magnitude <= 1.0:
                logger.warning(
                    f"Invalid impact_magnitude {impact_magnitude}, clamping to [0, 1]"
                )
                impact_magnitude = max(0.0, min(1.0, impact_magnitude))
            
            # Validate event_type
            event_type = str(data["event_type"]).lower()
            valid_event_types = ["earnings", "merger", "regulatory", "product_launch", "general"]
            if event_type not in valid_event_types:
                logger.warning(f"Unknown event_type {event_type}, defaulting to 'general'")
                event_type = "general"
            
            # Validate and convert confidence
            confidence = float(data["confidence"])
            if not 0.0 <= confidence <= 1.0:
                logger.warning(f"Invalid confidence {confidence}, clamping to [0, 1]")
                confidence = max(0.0, min(1.0, confidence))
            
            # Convert estimated_price_move
            estimated_price_move = float(data["estimated_price_move"])
            
            return StructuredAnalysis(
                sentiment=sentiment,  # type: ignore[arg-type]
                impact_magnitude=impact_magnitude,
                event_type=event_type,
                confidence=confidence,
                estimated_price_move=estimated_price_move,
            )
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to validate structured data: {e}")
            raise NLPAnalysisError(f"Data validation failed: {e}") from e

    def compute_news_score(self, analysis: StructuredAnalysis) -> float:
        """Compute aggregated news score from structured analysis.
        
        Uses nonlinear aggregation with confidence penalty and event type weighting.
        Score is clamped to [-1, 1] range.
        
        Formula:
        base_score = sentiment * impact_magnitude * estimated_price_move
        confidence_penalty = confidence ** 2  (quadratic penalty for low confidence)
        event_weight = EVENT_TYPE_WEIGHTS.get(event_type, 1.0)
        news_score = base_score * confidence_penalty * event_weight
        news_score = clamp(news_score, -1.0, 1.0)
        
        Args:
            analysis: Structured analysis data
            
        Returns:
            News score in range [-1, 1]
        """
        try:
            # Base score: sentiment * impact * estimated price move
            base_score = (
                analysis.sentiment
                * analysis.impact_magnitude
                * analysis.estimated_price_move
            )
            
            # Confidence penalty (quadratic) - low confidence reduces score
            confidence_penalty = analysis.confidence ** 2
            
            # Event type weighting from configuration
            event_weight = settings.nlp_event_type_weights.get(
                analysis.event_type.lower(), 1.0
            )
            
            # Compute final score
            news_score = base_score * confidence_penalty * event_weight
            
            # Clamp to [-1, 1] range
            news_score = float(np.clip(news_score, -1.0, 1.0))
            
            logger.debug(
                f"Computed news score: base={base_score:.3f}, "
                f"confidence_penalty={confidence_penalty:.3f}, "
                f"event_weight={event_weight:.2f}, "
                f"final={news_score:.3f}"
            )
            
            return news_score
            
        except Exception as e:
            logger.error(f"Failed to compute news score: {e}")
            # Return neutral score on error
            return 0.0

    async def analyze_batch(
        self,
        article_ids: List[str],
        max_concurrent: int = 5,
    ) -> List[ArticleAnalysis]:
        """Analyze multiple articles in batch.
        
        Processes articles with controlled concurrency to avoid
        overwhelming the Ollama service.
        
        Args:
            article_ids: List of article IDs to analyze
            max_concurrent: Maximum concurrent analyses
            
        Returns:
            List of article analyses
            
        Raises:
            NLPAnalysisError: If batch analysis fails
        """
        import asyncio
        
        logger.info(f"Starting batch analysis for {len(article_ids)} articles")
        
        results: List[ArticleAnalysis] = []
        errors: List[tuple[str, str]] = []
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(article_id: str) -> Optional[ArticleAnalysis]:
            """Analyze single article with semaphore control."""
            async with semaphore:
                try:
                    analysis = await self.analyze_article(article_id)
                    return analysis
                except Exception as e:
                    logger.error(f"Failed to analyze article {article_id}: {e}")
                    errors.append((article_id, str(e)))
                    return None
        
        # Create tasks for all articles
        tasks = [analyze_with_semaphore(article_id) for article_id in article_ids]
        
        # Execute with progress tracking
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                results.append(result)
            completed += 1
            
            if completed % 10 == 0 or completed == len(article_ids):
                logger.info(f"Batch analysis progress: {completed}/{len(article_ids)}")
        
        logger.info(
            f"Batch analysis completed: {len(results)} successful, {len(errors)} failed"
        )
        
        if errors:
            logger.warning(f"Errors during batch analysis: {errors[:5]}")  # Log first 5
        
        return results

    async def analyze_latest_unanalyzed(
        self,
        limit: int = 100,
        max_concurrent: int = 5,
    ) -> List[ArticleAnalysis]:
        """Analyze latest unanalyzed articles.
        
        Convenience method that fetches unanalyzed articles and processes them.
        
        Args:
            limit: Maximum number of articles to analyze
            max_concurrent: Maximum concurrent analyses
            
        Returns:
            List of article analyses
            
        Raises:
            NLPAnalysisError: If analysis fails
        """
        try:
            # Get unanalyzed articles
            articles = await self.article_repo.get_unanalyzed(limit=limit)
            
            if not articles:
                logger.info("No unanalyzed articles found")
                return []
            
            logger.info(f"Found {len(articles)} unanalyzed articles")
            
            # Extract article IDs
            article_ids = [article.id for article in articles]
            
            # Analyze in batch
            return await self.analyze_batch(article_ids, max_concurrent=max_concurrent)
            
        except Exception as e:
            logger.error(f"Failed to analyze latest unanalyzed articles: {e}")
            raise NLPAnalysisError(f"Batch analysis failed: {e}") from e
