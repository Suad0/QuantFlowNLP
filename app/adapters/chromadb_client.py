"""ChromaDB client adapter for vector storage operations.

This module provides an async wrapper around ChromaDB for storing and querying
article embeddings with persistent storage support.
"""

import asyncio
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.exceptions import VectorRepositoryError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ChromaDBClient:
    """Async wrapper for ChromaDB operations.
    
    Provides connection management, collection handling, and async operations
    for storing and querying vector embeddings.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        persist_directory: Optional[str] = None,
    ):
        """Initialize ChromaDB client.
        
        Args:
            host: ChromaDB host (defaults to settings)
            port: ChromaDB port (defaults to settings)
            persist_directory: Directory for persistent storage (defaults to settings)
        """
        self.host = host or settings.chromadb_host
        self.port = port or settings.chromadb_port
        self.persist_directory = persist_directory or settings.chromadb_persist_directory
        
        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None
        self._collection_name = settings.chromadb_collection

    async def connect(self) -> None:
        """Establish connection to ChromaDB.
        
        Creates or connects to the ChromaDB instance with persistent storage.
        
        Raises:
            VectorRepositoryError: If connection fails
        """
        try:
            logger.info(
                "Connecting to ChromaDB",
                extra={
                    "host": self.host,
                    "port": self.port,
                    "persist_directory": self.persist_directory,
                },
            )
            
            # Run blocking ChromaDB operations in thread pool
            self._client = await asyncio.to_thread(
                chromadb.HttpClient,
                host=self.host,
                port=self.port,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=False,
                ),
            )
            
            logger.info("ChromaDB client connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise VectorRepositoryError(f"ChromaDB connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from ChromaDB.
        
        Cleans up resources and closes connections.
        """
        if self._client:
            logger.info("Disconnecting from ChromaDB")
            self._collection = None
            self._client = None

    async def get_or_create_collection(
        self,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> chromadb.Collection:
        """Get or create a ChromaDB collection.
        
        Args:
            name: Collection name (defaults to configured collection)
            metadata: Collection metadata for configuration
            
        Returns:
            ChromaDB collection instance
            
        Raises:
            VectorRepositoryError: If collection creation fails
        """
        if not self._client:
            await self.connect()
        
        collection_name = name or self._collection_name
        
        try:
            # Default metadata for HNSW index configuration
            default_metadata = {
                "hnsw:space": "cosine",  # Cosine similarity
                "hnsw:construction_ef": 200,  # Construction time accuracy
                "hnsw:M": 16,  # Number of connections per layer
            }
            
            if metadata:
                default_metadata.update(metadata)
            
            logger.info(
                f"Getting or creating collection: {collection_name}",
                extra={"metadata": default_metadata},
            )
            
            self._collection = await asyncio.to_thread(
                self._client.get_or_create_collection,  # type: ignore[union-attr]
                name=collection_name,
                metadata=default_metadata,
            )
            
            logger.info(f"Collection '{collection_name}' ready")
            return self._collection
            
        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}")
            raise VectorRepositoryError(f"Collection operation failed: {e}") from e

    async def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """Add embeddings to the collection.
        
        Args:
            ids: List of unique identifiers
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts
            documents: Optional list of document texts
            
        Raises:
            VectorRepositoryError: If add operation fails
        """
        if not self._collection:
            await self.get_or_create_collection()
        
        try:
            logger.debug(
                f"Adding {len(ids)} embeddings to collection",
                extra={"collection": self._collection_name},
            )
            
            await asyncio.to_thread(
                self._collection.add,  # type: ignore[union-attr]
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            
            logger.debug(f"Successfully added {len(ids)} embeddings")
            
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            raise VectorRepositoryError(f"Failed to add embeddings: {e}") from e

    async def query_similar(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query for similar embeddings.
        
        Args:
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return per query
            where: Metadata filter conditions
            where_document: Document filter conditions
            
        Returns:
            Query results with ids, distances, metadatas, and documents
            
        Raises:
            VectorRepositoryError: If query fails
        """
        if not self._collection:
            await self.get_or_create_collection()
        
        try:
            logger.debug(
                f"Querying {len(query_embeddings)} embeddings",
                extra={"n_results": n_results, "where": where},
            )
            
            results = await asyncio.to_thread(
                self._collection.query,  # type: ignore[union-attr]
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document,
            )
            
            logger.debug(f"Query returned results")
            return results  # type: ignore[no-any-return]
            
        except Exception as e:
            logger.error(f"Failed to query embeddings: {e}")
            raise VectorRepositoryError(f"Failed to query embeddings: {e}") from e

    async def get_by_ids(
        self,
        ids: List[str],
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get embeddings by IDs.
        
        Args:
            ids: List of embedding IDs to retrieve
            include: List of fields to include (embeddings, metadatas, documents)
            
        Returns:
            Retrieved embeddings and metadata
            
        Raises:
            VectorRepositoryError: If get operation fails
        """
        if not self._collection:
            await self.get_or_create_collection()
        
        try:
            logger.debug(f"Getting {len(ids)} embeddings by ID")
            
            results = await asyncio.to_thread(
                self._collection.get,  # type: ignore[union-attr]
                ids=ids,
                include=include or ["embeddings", "metadatas", "documents"],
            )
            
            return results  # type: ignore[no-any-return]
            
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            raise VectorRepositoryError(f"Failed to get embeddings: {e}") from e

    async def delete_by_ids(self, ids: List[str]) -> None:
        """Delete embeddings by IDs.
        
        Args:
            ids: List of embedding IDs to delete
            
        Raises:
            VectorRepositoryError: If delete operation fails
        """
        if not self._collection:
            await self.get_or_create_collection()
        
        try:
            logger.debug(f"Deleting {len(ids)} embeddings")
            
            await asyncio.to_thread(
                self._collection.delete,  # type: ignore[union-attr]
                ids=ids,
            )
            
            logger.debug(f"Successfully deleted {len(ids)} embeddings")
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            raise VectorRepositoryError(f"Failed to delete embeddings: {e}") from e

    async def count(self) -> int:
        """Get count of embeddings in collection.
        
        Returns:
            Number of embeddings in collection
            
        Raises:
            VectorRepositoryError: If count operation fails
        """
        if not self._collection:
            await self.get_or_create_collection()
        
        try:
            count = await asyncio.to_thread(
                self._collection.count  # type: ignore[union-attr]
            )
            return count  # type: ignore[no-any-return]
            
        except Exception as e:
            logger.error(f"Failed to count embeddings: {e}")
            raise VectorRepositoryError(f"Failed to count embeddings: {e}") from e

    async def health_check(self) -> bool:
        """Check if ChromaDB is healthy and accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self._client:
                await self.connect()
            
            # Try to get heartbeat
            await asyncio.to_thread(self._client.heartbeat)  # type: ignore[union-attr]
            return True
            
        except Exception as e:
            logger.warning(f"ChromaDB health check failed: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        """Check if client is connected.
        
        Returns:
            True if connected, False otherwise
        """
        return self._client is not None

    @property
    def collection_name(self) -> str:
        """Get current collection name.
        
        Returns:
            Collection name
        """
        return self._collection_name
