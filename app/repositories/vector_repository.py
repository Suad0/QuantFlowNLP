"""Vector repository implementation using ChromaDB.

This module provides the concrete implementation of the VectorRepository interface
for storing and querying article embeddings with deduplication support.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.adapters.chromadb_client import ChromaDBClient
from app.core.exceptions import VectorRepositoryError
from app.repositories.base import VectorRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ChromaVectorRepository(VectorRepository):
    """ChromaDB implementation of VectorRepository.
    
    Provides vector storage and similarity search capabilities using ChromaDB
    for article embedding deduplication and semantic search.
    """

    def __init__(self, chroma_client: ChromaDBClient):
        """Initialize repository with ChromaDB client.
        
        Args:
            chroma_client: ChromaDB client instance
        """
        self.client = chroma_client

    async def store_embedding(
        self,
        article_id: str,
        embedding: List[float],
        metadata: dict,
    ) -> None:
        """Store an article embedding with metadata.
        
        Args:
            article_id: Unique article identifier
            embedding: Embedding vector
            metadata: Additional metadata (source, title, published_at, etc.)
            
        Raises:
            VectorRepositoryError: If storage fails
        """
        try:
            logger.debug(
                f"Storing embedding for article {article_id}",
                extra={"metadata": metadata},
            )
            
            # Ensure collection exists
            await self.client.get_or_create_collection()
            
            # Prepare metadata - ChromaDB requires string values
            chroma_metadata = self._prepare_metadata(metadata)
            
            # Store embedding
            await self.client.add_embeddings(
                ids=[article_id],
                embeddings=[embedding],
                metadatas=[chroma_metadata],
            )
            
            logger.info(f"Successfully stored embedding for article {article_id}")
            
        except Exception as e:
            logger.error(f"Failed to store embedding for article {article_id}: {e}")
            raise VectorRepositoryError(
                f"Failed to store embedding for article {article_id}: {e}"
            ) from e

    async def find_similar(
        self,
        embedding: List[float],
        threshold: float = 0.95,
        limit: int = 10,
    ) -> List[dict]:
        """Find similar embeddings above a similarity threshold.
        
        Uses cosine similarity to find articles with similar embeddings.
        Similarity scores are converted from distances (lower is more similar)
        to similarity scores (higher is more similar).
        
        Args:
            embedding: Query embedding vector
            threshold: Minimum similarity threshold (0-1)
            limit: Maximum number of results
            
        Returns:
            List of similar articles with similarity scores and metadata
            
        Raises:
            VectorRepositoryError: If query fails
        """
        try:
            logger.debug(
                f"Searching for similar embeddings",
                extra={"threshold": threshold, "limit": limit},
            )
            
            # Ensure collection exists
            await self.client.get_or_create_collection()
            
            # Query for similar embeddings
            results = await self.client.query_similar(
                query_embeddings=[embedding],
                n_results=limit,
            )
            
            # Process results
            similar_articles = []
            
            if results and "ids" in results and len(results["ids"]) > 0:
                ids = results["ids"][0]
                distances = results.get("distances", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                
                for i, article_id in enumerate(ids):
                    # Convert distance to similarity score
                    # For cosine distance: similarity = 1 - distance
                    distance = distances[i] if i < len(distances) else 1.0
                    similarity = 1.0 - distance
                    
                    # Filter by threshold
                    if similarity >= threshold:
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        
                        similar_articles.append({
                            "article_id": article_id,
                            "similarity": similarity,
                            "distance": distance,
                            "metadata": self._restore_metadata(metadata),
                        })
            
            logger.debug(
                f"Found {len(similar_articles)} similar articles above threshold"
            )
            
            return similar_articles
            
        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {e}")
            raise VectorRepositoryError(
                f"Failed to find similar embeddings: {e}"
            ) from e

    async def get_by_article_id(self, article_id: str) -> Optional[dict]:
        """Get embedding by article ID.
        
        Args:
            article_id: Article identifier
            
        Returns:
            Embedding data with metadata if found, None otherwise
            
        Raises:
            VectorRepositoryError: If retrieval fails
        """
        try:
            logger.debug(f"Getting embedding for article {article_id}")
            
            # Ensure collection exists
            await self.client.get_or_create_collection()
            
            # Get embedding by ID
            results = await self.client.get_by_ids(
                ids=[article_id],
                include=["embeddings", "metadatas"],
            )
            
            if results and "ids" in results and len(results["ids"]) > 0:
                embeddings = results.get("embeddings", [])
                metadatas = results.get("metadatas", [])
                
                if embeddings and metadatas:
                    return {
                        "article_id": article_id,
                        "embedding": embeddings[0],
                        "metadata": self._restore_metadata(metadatas[0]),
                    }
            
            logger.debug(f"No embedding found for article {article_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get embedding for article {article_id}: {e}")
            raise VectorRepositoryError(
                f"Failed to get embedding for article {article_id}: {e}"
            ) from e

    async def delete_by_article_id(self, article_id: str) -> None:
        """Delete embedding by article ID.
        
        Args:
            article_id: Article identifier
            
        Raises:
            VectorRepositoryError: If deletion fails
        """
        try:
            logger.debug(f"Deleting embedding for article {article_id}")
            
            # Ensure collection exists
            await self.client.get_or_create_collection()
            
            # Delete embedding
            await self.client.delete_by_ids(ids=[article_id])
            
            logger.info(f"Successfully deleted embedding for article {article_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete embedding for article {article_id}: {e}")
            raise VectorRepositoryError(
                f"Failed to delete embedding for article {article_id}: {e}"
            ) from e

    async def count(self) -> int:
        """Get total count of stored embeddings.
        
        Returns:
            Number of embeddings in storage
            
        Raises:
            VectorRepositoryError: If count fails
        """
        try:
            # Ensure collection exists
            await self.client.get_or_create_collection()
            
            count = await self.client.count()
            logger.debug(f"Total embeddings in storage: {count}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to count embeddings: {e}")
            raise VectorRepositoryError(f"Failed to count embeddings: {e}") from e

    async def check_duplicate(
        self,
        embedding: List[float],
        threshold: float = 0.95,
    ) -> Optional[str]:
        """Check if an embedding is a duplicate of an existing one.
        
        This is a convenience method for deduplication that returns the ID
        of the most similar article if it exceeds the threshold.
        
        Args:
            embedding: Query embedding vector
            threshold: Minimum similarity threshold for duplicate detection
            
        Returns:
            Article ID of duplicate if found, None otherwise
            
        Raises:
            VectorRepositoryError: If check fails
        """
        try:
            similar = await self.find_similar(
                embedding=embedding,
                threshold=threshold,
                limit=1,
            )
            
            if similar:
                duplicate_id = similar[0]["article_id"]
                similarity = similar[0]["similarity"]
                
                logger.info(
                    f"Duplicate detected: {duplicate_id} (similarity: {similarity:.4f})"
                )
                return duplicate_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check for duplicates: {e}")
            raise VectorRepositoryError(
                f"Failed to check for duplicates: {e}"
            ) from e

    def _prepare_metadata(self, metadata: dict) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB storage.
        
        ChromaDB requires metadata values to be strings, ints, floats, or bools.
        This method converts datetime objects and other types to strings.
        
        Args:
            metadata: Original metadata dict
            
        Returns:
            Prepared metadata dict
        """
        prepared = {}
        
        for key, value in metadata.items():
            if isinstance(value, datetime):
                prepared[key] = value.isoformat()
            elif isinstance(value, (str, int, float, bool)):
                prepared[key] = value
            elif value is None:
                prepared[key] = ""
            else:
                # Convert other types to string
                prepared[key] = str(value)
        
        return prepared

    def _restore_metadata(self, metadata: dict) -> dict:
        """Restore metadata from ChromaDB storage.
        
        Converts ISO format datetime strings back to datetime objects.
        
        Args:
            metadata: Stored metadata dict
            
        Returns:
            Restored metadata dict
        """
        restored = {}
        
        for key, value in metadata.items():
            # Try to parse datetime strings
            if isinstance(value, str) and "T" in value:
                try:
                    restored[key] = datetime.fromisoformat(value)
                except (ValueError, TypeError):
                    restored[key] = value
            else:
                restored[key] = value
        
        return restored
