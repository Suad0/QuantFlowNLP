"""Unit tests for Ollama client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.adapters.ollama_client import OllamaClient
from app.core.exceptions import CircuitBreakerOpenError, OllamaConnectionError


@pytest.fixture
def ollama_client():
    """Create Ollama client for testing."""
    client = OllamaClient(base_url="http://localhost:11434", timeout=30)
    yield client


@pytest.mark.asyncio
async def test_ollama_client_initialization(ollama_client):
    """Test Ollama client initialization."""
    assert ollama_client.base_url == "http://localhost:11434"
    assert ollama_client.timeout == 30
    assert ollama_client.client is not None
    assert ollama_client.circuit_breaker is not None


@pytest.mark.asyncio
async def test_generate_success(ollama_client):
    """Test successful text generation."""
    mock_response = {"response": "This is a generated response."}

    with patch.object(ollama_client, "_make_request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response

        result = await ollama_client.generate(
            model="llama3",
            prompt="Test prompt",
            temperature=0.7,
        )

        assert result == "This is a generated response."
        mock_request.assert_called_once()


@pytest.mark.asyncio
async def test_generate_with_system_prompt(ollama_client):
    """Test text generation with system prompt."""
    mock_response = {"response": "Response with system context."}

    with patch.object(ollama_client, "_make_request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response

        result = await ollama_client.generate(
            model="llama3",
            prompt="Test prompt",
            system="You are a helpful assistant.",
            temperature=0.5,
            max_tokens=100,
        )

        assert result == "Response with system context."
        call_args = mock_request.call_args
        assert call_args[0][2]["system"] == "You are a helpful assistant."
        assert call_args[0][2]["options"]["temperature"] == 0.5
        assert call_args[0][2]["options"]["num_predict"] == 100


@pytest.mark.asyncio
async def test_embed_success(ollama_client):
    """Test successful embedding generation."""
    mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_response = {"embedding": mock_embedding}

    with patch.object(ollama_client, "_make_request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response

        result = await ollama_client.embed(
            model="nomic-embed-text",
            text="Test text for embedding",
        )

        assert result == mock_embedding
        mock_request.assert_called_once()


@pytest.mark.asyncio
async def test_embed_batch(ollama_client):
    """Test batch embedding generation."""
    texts = ["Text 1", "Text 2", "Text 3"]
    mock_embedding = [0.1, 0.2, 0.3]

    with patch.object(ollama_client, "embed", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = mock_embedding

        results = await ollama_client.embed_batch(
            model="nomic-embed-text",
            texts=texts,
            batch_size=2,
        )

        assert len(results) == 3
        assert all(emb == mock_embedding for emb in results)
        assert mock_embed.call_count == 3


@pytest.mark.asyncio
async def test_health_check_success(ollama_client):
    """Test successful health check."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"models": [{"name": "llama3"}]}
    mock_response.raise_for_status = MagicMock()

    with patch.object(ollama_client.client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        result = await ollama_client.health_check()

        assert result is True
        mock_get.assert_called_once_with("/api/tags")


@pytest.mark.asyncio
async def test_health_check_failure(ollama_client):
    """Test health check failure."""
    with patch.object(ollama_client.client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = Exception("Connection failed")

        result = await ollama_client.health_check()

        assert result is False


@pytest.mark.asyncio
async def test_list_models(ollama_client):
    """Test listing available models."""
    mock_models = [
        {"name": "llama3", "size": 1000000},
        {"name": "nomic-embed-text", "size": 500000},
    ]
    mock_response = {"models": mock_models}

    with patch.object(ollama_client, "_make_request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response

        result = await ollama_client.list_models()

        assert result == mock_models
        mock_request.assert_called_once_with("GET", "/api/tags")


@pytest.mark.asyncio
async def test_pull_model(ollama_client):
    """Test pulling a model."""
    mock_response = {"status": "success"}

    with patch.object(ollama_client, "_make_request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response

        result = await ollama_client.pull_model("llama3")

        assert result == mock_response
        mock_request.assert_called_once_with("POST", "/api/pull", {"name": "llama3"})


@pytest.mark.asyncio
async def test_circuit_breaker_stats(ollama_client):
    """Test getting circuit breaker statistics."""
    stats = ollama_client.get_circuit_breaker_stats()

    assert "state" in stats
    assert "failure_count" in stats
    assert "failure_threshold" in stats
    assert stats["state"] == "closed"


@pytest.mark.asyncio
async def test_circuit_breaker_reset(ollama_client):
    """Test resetting circuit breaker."""
    # Manually set failure count
    ollama_client.circuit_breaker.failure_count = 5

    ollama_client.reset_circuit_breaker()

    stats = ollama_client.get_circuit_breaker_stats()
    assert stats["failure_count"] == 0
    assert stats["state"] == "closed"


@pytest.mark.asyncio
async def test_context_manager(ollama_client):
    """Test using Ollama client as async context manager."""
    async with OllamaClient() as client:
        assert client.client is not None

    # Client should be closed after context exit
    # Note: We can't easily test this without accessing internal state
