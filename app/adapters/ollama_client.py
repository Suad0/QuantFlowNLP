"""Async Ollama HTTP client for LLM inference and embeddings.

This module provides an async client for interacting with the Ollama API,
including text generation, embedding generation, and health checks.
"""

import logging
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import OllamaConnectionError
from app.utils.circuit_breaker import CircuitBreaker
from app.utils.retry import retry_async

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async client for Ollama API.

    Provides methods for text generation, embedding generation, and health checks
    with built-in retry logic and circuit breaker protection.

    Attributes:
        base_url: Ollama API base URL
        timeout: Request timeout in seconds
        client: Async HTTP client
        circuit_breaker: Circuit breaker for failure protection
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ):
        """Initialize Ollama client.

        Args:
            base_url: Ollama API base URL (default from settings)
            timeout: Request timeout in seconds (default from settings)
            max_retries: Maximum retry attempts (default from settings)
        """
        self.base_url = base_url or settings.ollama_base_url
        self.timeout = timeout or settings.ollama_timeout
        self.max_retries = max_retries or settings.ollama_max_retries

        # Create async HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
            ),
        )

        # Initialize circuit breaker for failure protection
        self.circuit_breaker = CircuitBreaker()

        logger.info(f"Initialized OllamaClient with base_url={self.base_url}")

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        await self.client.aclose()
        logger.info("OllamaClient closed")

    async def __aenter__(self) -> "OllamaClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    @retry_async(exceptions=(httpx.RequestError, httpx.TimeoutException))
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request to Ollama API with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            json_data: JSON request body

        Returns:
            JSON response as dictionary

        Raises:
            OllamaConnectionError: If request fails after retries
        """
        try:
            response = await self.client.request(
                method=method,
                url=endpoint,
                json=json_data,
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API returned error status: {e.response.status_code}")
            raise OllamaConnectionError(
                f"Ollama API error: {e.response.status_code} - {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            logger.error(f"Ollama API request failed: {e}")
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in Ollama request: {e}")
            raise OllamaConnectionError(f"Unexpected Ollama error: {e}") from e

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """Generate text using Ollama LLM.

        Args:
            model: Model name (e.g., "llama3")
            prompt: Input prompt for generation
            system: System prompt for context
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response (not implemented)
            **kwargs: Additional parameters for Ollama API

        Returns:
            Generated text

        Raises:
            OllamaConnectionError: If generation fails
        """
        logger.debug(f"Generating text with model={model}, prompt_length={len(prompt)}")

        # Build request payload
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }

        if system:
            payload["system"] = system

        if temperature is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["temperature"] = temperature

        if max_tokens is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["num_predict"] = max_tokens

        # Add any additional kwargs
        payload.update(kwargs)

        # Make request with circuit breaker protection
        try:
            response = await self.circuit_breaker.call(
                self._make_request,
                "POST",
                "/api/generate",
                payload,
            )

            generated_text = response.get("response", "")
            logger.debug(f"Generated text length: {len(generated_text)}")

            return generated_text
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise

    async def embed(
        self,
        model: str,
        text: str,
        **kwargs: Any,
    ) -> list[float]:
        """Generate embedding for text using Ollama.

        Args:
            model: Embedding model name (e.g., "nomic-embed-text")
            text: Input text to embed
            **kwargs: Additional parameters for Ollama API

        Returns:
            Embedding vector as list of floats

        Raises:
            OllamaConnectionError: If embedding generation fails
        """
        logger.debug(f"Generating embedding with model={model}, text_length={len(text)}")

        # Build request payload
        payload: dict[str, Any] = {
            "model": model,
            "prompt": text,
        }

        # Add any additional kwargs
        payload.update(kwargs)

        # Make request with circuit breaker protection
        try:
            response = await self.circuit_breaker.call(
                self._make_request,
                "POST",
                "/api/embeddings",
                payload,
            )

            embedding = response.get("embedding", [])
            logger.debug(f"Generated embedding dimension: {len(embedding)}")

            return embedding  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def embed_batch(
        self,
        model: str,
        texts: list[str],
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts in batches.

        Args:
            model: Embedding model name
            texts: List of texts to embed
            batch_size: Batch size for processing (default from settings)
            **kwargs: Additional parameters for Ollama API

        Returns:
            List of embedding vectors

        Raises:
            OllamaConnectionError: If embedding generation fails
        """
        _batch_size = batch_size or settings.news_embedding_batch_size
        embeddings: list[list[float]] = []

        logger.info(f"Generating embeddings for {len(texts)} texts in batches of {_batch_size}")

        for i in range(0, len(texts), _batch_size):
            batch = texts[i : i + _batch_size]
            logger.debug(f"Processing batch {i // _batch_size + 1}/{(len(texts) + _batch_size - 1) // _batch_size}")

            for text in batch:
                embedding = await self.embed(model, text, **kwargs)
                embeddings.append(embedding)

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    async def health_check(self) -> bool:
        """Check if Ollama service is available and healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Try to list available models as a health check
            response = await self.client.get("/api/tags")
            response.raise_for_status()

            models = response.json().get("models", [])
            logger.info(f"Ollama health check passed. Available models: {len(models)}")

            return True
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models in Ollama.

        Returns:
            List of model information dictionaries

        Raises:
            OllamaConnectionError: If request fails
        """
        try:
            response = await self._make_request("GET", "/api/tags")
            models = response.get("models", [])
            logger.info(f"Retrieved {len(models)} available models")
            return models  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise

    async def pull_model(self, model: str) -> dict[str, Any]:
        """Pull/download a model from Ollama registry.

        Args:
            model: Model name to pull

        Returns:
            Response from pull operation

        Raises:
            OllamaConnectionError: If pull fails
        """
        logger.info(f"Pulling model: {model}")

        try:
            response = await self._make_request(
                "POST",
                "/api/pull",
                {"name": model},
            )
            logger.info(f"Successfully pulled model: {model}")
            return response
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            raise

    def get_circuit_breaker_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with circuit breaker stats
        """
        return self.circuit_breaker.get_stats()

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self.circuit_breaker.reset()
        logger.info("Circuit breaker manually reset")
