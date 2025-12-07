"""Unit tests for VectorRepository.

Tests the ChromaDB implementation of vector storage and similarity search.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.adapters.chromadb_client import ChromaDBClient
from app.core.exceptions import VectorRepositoryError
from app.repositories.vector_repository import ChromaVectorRepository


class MockChromaCollection:
    """Mock ChromaDB collection for testing."""

    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, Any]] = {}

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str] = None,
    ) -> None:
        """Mock add operation."""
        for i, id_ in enumerate(ids):
            self.data[id_] = {
                "embedding": embeddings[i],
                "metadata": metadatas[i] if i < len(metadatas) else {},
            }

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Dict[str, Any] = None,
        where_document: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Mock query operation with cosine similarity."""
        if not self.data:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}

        query_embedding = query_embeddings[0]
        results = []

        for id_, item in self.data.items():
            # Calculate cosine distance (simplified)
            embedding = item["embedding"]
            distance = self._cosine_distance(query_embedding, embedding)
            results.append((id_, distance, item["metadata"]))

        # Sort by distance (lower is more similar)
        results.sort(key=lambda x: x[1])
        results = results[:n_results]

        ids = [r[0] for r in results]
        distances = [r[1] for r in results]
        metadatas = [r[2] for r in results]

        return {
            "ids": [ids],
            "distances": [distances],
            "metadatas": [metadatas],
        }

    def get(
        self,
        ids: List[str],
        include: List[str] = None,
    ) -> Dict[str, Any]:
        """Mock get operation."""
        result_ids = []
        result_embeddings = []
        result_metadatas = []

        for id_ in ids:
            if id_ in self.data:
                result_ids.append(id_)
                result_embeddings.append(self.data[id_]["embedding"])
                result_metadatas.append(self.data[id_]["metadata"])

        return {
            "ids": result_ids,
            "embeddings": result_embeddings,
            "metadatas": result_metadatas,
        }

    def delete(self, ids: List[str]) -> None:
        """Mock delete operation."""
        for id_ in ids:
            if id_ in self.data:
                del self.data[id_]

    def count(self) -> int:
        """Mock count operation."""
        return len(self.data)

    @staticmethod
    def _cosine_distance(a: List[float], b: List[float]) -> float:
        """Calculate cosine distance between two vectors."""
        # Simplified calculation for testing
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 1.0
        
        similarity = dot_product / (norm_a * norm_b)
        return 1.0 - similarity  # Convert to distance


@pytest.fixture
async def mock_chroma_client():
    """Create a mock ChromaDB client for testing."""
    client = ChromaDBClient()
    client._client = MagicMock()
    client._collection = MockChromaCollection()
    
    # Mock async operations
    async def mock_get_or_create_collection(*args, **kwargs):
        return client._collection
    
    client.get_or_create_collection = mock_get_or_create_collection
    
    return client


@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector."""
    return [0.1] * 768  # 768-dimensional embedding


@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    return {
        "source": "reuters",
        "title": "Test Article",
        "published_at": datetime.now(timezone.utc),
        "url": "https://example.com/article/123",
    }


@pytest.mark.asyncio
async def test_store_embedding(mock_chroma_client, sample_embedding, sample_metadata):
    """Test storing an embedding."""
    repo = ChromaVectorRepository(mock_chroma_client)
    article_id = str(uuid.uuid4())
    
    await repo.store_embedding(article_id, sample_embedding, sample_metadata)
    
    # Verify embedding was stored
    result = await repo.get_by_article_id(article_id)
    assert result is not None
    assert result["article_id"] == article_id
    assert len(result["embedding"]) == 768


@pytest.mark.asyncio
async def test_find_similar_above_threshold(mock_chroma_client, sample_embedding, sample_metadata):
    """Test finding similar embeddings above threshold."""
    repo = ChromaVectorRepository(mock_chroma_client)
    
    # Store a few embeddings
    article_id1 = str(uuid.uuid4())
    article_id2 = str(uuid.uuid4())
    
    await repo.store_embedding(article_id1, sample_embedding, sample_metadata)
    
    # Store a very similar embedding (same values)
    await repo.store_embedding(article_id2, sample_embedding, sample_metadata)
    
    # Find similar embeddings
    similar = await repo.find_similar(sample_embedding, threshold=0.95, limit=10)
    
    # Should find both articles (they're identical)
    assert len(similar) >= 1
    assert all(s["similarity"] >= 0.95 for s in similar)


@pytest.mark.asyncio
async def test_find_similar_below_threshold(mock_chroma_client, sample_embedding, sample_metadata):
    """Test that dissimilar embeddings are filtered out."""
    repo = ChromaVectorRepository(mock_chroma_client)
    
    # Store an embedding
    article_id = str(uuid.uuid4())
    await repo.store_embedding(article_id, sample_embedding, sample_metadata)
    
    # Query with a very different embedding
    different_embedding = [-0.1] * 768
    similar = await repo.find_similar(different_embedding, threshold=0.95, limit=10)
    
    # Should not find any similar articles above threshold
    assert len(similar) == 0


@pytest.mark.asyncio
async def test_get_by_article_id(mock_chroma_client, sample_embedding, sample_metadata):
    """Test retrieving embedding by article ID."""
    repo = ChromaVectorRepository(mock_chroma_client)
    article_id = str(uuid.uuid4())
    
    await repo.store_embedding(article_id, sample_embedding, sample_metadata)
    
    result = await repo.get_by_article_id(article_id)
    
    assert result is not None
    assert result["article_id"] == article_id
    assert result["embedding"] == sample_embedding
    assert "metadata" in result


@pytest.mark.asyncio
async def test_get_by_article_id_not_found(mock_chroma_client):
    """Test retrieving non-existent embedding."""
    repo = ChromaVectorRepository(mock_chroma_client)
    
    result = await repo.get_by_article_id("nonexistent-id")
    
    assert result is None


@pytest.mark.asyncio
async def test_delete_by_article_id(mock_chroma_client, sample_embedding, sample_metadata):
    """Test deleting an embedding."""
    repo = ChromaVectorRepository(mock_chroma_client)
    article_id = str(uuid.uuid4())
    
    await repo.store_embedding(article_id, sample_embedding, sample_metadata)
    
    # Verify it exists
    result = await repo.get_by_article_id(article_id)
    assert result is not None
    
    # Delete it
    await repo.delete_by_article_id(article_id)
    
    # Verify it's gone
    result = await repo.get_by_article_id(article_id)
    assert result is None


@pytest.mark.asyncio
async def test_count(mock_chroma_client, sample_embedding, sample_metadata):
    """Test counting embeddings."""
    repo = ChromaVectorRepository(mock_chroma_client)
    
    initial_count = await repo.count()
    assert initial_count == 0
    
    # Add embeddings
    for i in range(3):
        article_id = str(uuid.uuid4())
        await repo.store_embedding(article_id, sample_embedding, sample_metadata)
    
    final_count = await repo.count()
    assert final_count == 3


@pytest.mark.asyncio
async def test_check_duplicate_found(mock_chroma_client, sample_embedding, sample_metadata):
    """Test duplicate detection when duplicate exists."""
    repo = ChromaVectorRepository(mock_chroma_client)
    article_id = str(uuid.uuid4())
    
    await repo.store_embedding(article_id, sample_embedding, sample_metadata)
    
    # Check for duplicate with same embedding
    duplicate_id = await repo.check_duplicate(sample_embedding, threshold=0.95)
    
    assert duplicate_id is not None
    assert duplicate_id == article_id


@pytest.mark.asyncio
async def test_check_duplicate_not_found(mock_chroma_client, sample_embedding, sample_metadata):
    """Test duplicate detection when no duplicate exists."""
    repo = ChromaVectorRepository(mock_chroma_client)
    article_id = str(uuid.uuid4())
    
    await repo.store_embedding(article_id, sample_embedding, sample_metadata)
    
    # Check for duplicate with different embedding
    different_embedding = [-0.1] * 768
    duplicate_id = await repo.check_duplicate(different_embedding, threshold=0.95)
    
    assert duplicate_id is None


@pytest.mark.asyncio
async def test_metadata_datetime_conversion(mock_chroma_client, sample_embedding):
    """Test that datetime objects are properly converted in metadata."""
    repo = ChromaVectorRepository(mock_chroma_client)
    article_id = str(uuid.uuid4())
    
    metadata = {
        "published_at": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "source": "reuters",
    }
    
    await repo.store_embedding(article_id, sample_embedding, metadata)
    
    result = await repo.get_by_article_id(article_id)
    
    assert result is not None
    assert "published_at" in result["metadata"]
    # Should be converted back to datetime
    assert isinstance(result["metadata"]["published_at"], datetime)
