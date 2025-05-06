"""
Database Manager Module

This module handles interactions with ChromaDB for vector storage and retrieval.
It provides a clean interface for managing document collections, including creation,
retrieval, and deletion of vector stores.

Key Components:
    - ChromaDB client initialization and management
    - Collection CRUD operations
    - Processed files tracking
    - Basic error handling for database operations
"""

import json
from typing import Any, Dict, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from rag.settings import Settings as AppSettings
from rag.utils import logger

class DBManager:
    """
    Manages ChromaDB operations for vector storage.
    
    This class handles:
    1. ChromaDB client initialization
    2. Collection management
    3. Document tracking
    4. Basic error recovery
    
    Attributes:
        client: ChromaDB client instance
        processed_files: Dictionary mapping filenames to collection IDs
    """
    
    def __init__(self):
        """
        Initialize database manager with ChromaDB client and load processed files.
        
        Sets up:
            - Directory structure
            - ChromaDB client
            - Processed files tracking
        """
        AppSettings.initialize_directories()
        self.client = self._initialize_client()
        self.processed_files = AppSettings.load_processed_files()
    
    def _initialize_client(self) -> chromadb.PersistentClient:
        """
        Initialize ChromaDB client with persistence.
        
        Returns:
            chromadb.PersistentClient: Configured ChromaDB client instance
            
        The client is configured with:
            - Persistence enabled
            - Telemetry disabled
            - Reset capability enabled
        """
        return chromadb.PersistentClient(
            path=AppSettings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    
    def _save_processed_files(self) -> None:
        """
        Save the mapping of processed files to collection IDs.
        
        This method persists the current state of processed files to disk,
        maintaining a record of all document collections.
        """
        with open(AppSettings.PROCESSED_FILES_PATH, 'w') as f:
            json.dump(self.processed_files, f, indent=2)
    
    def get_collection(self, collection_id: str) -> Optional[Any]:
        """
        Retrieve a collection by its ID.
        
        Args:
            collection_id: Unique identifier for the collection
            
        Returns:
            Collection object if found, None if not found or error occurs
            
        The method includes basic error handling and logging.
        """
        try:
            return self.client.get_collection(name=collection_id)
        except Exception as e:
            logger.error(f"Error retrieving collection {collection_id}: {e}")
            return None
    
    def create_collection(self, collection_id: str, metadata: Dict[str, Any]) -> Any:
        """
        Create a new collection for document storage.
        
        Args:
            collection_id: Unique identifier for the new collection
            metadata: Additional information about the collection
            
        Returns:
            Newly created collection object
            
        This method sets up a new vector store for document chunks with proper configuration
        to ensure embeddings are stored correctly.
        """
        try:
            collection = self.client.create_collection(
                name=collection_id,
                metadata=metadata or {},
                embedding_function=None
            )
            logger.info(f"Created collection: {collection_id}")
            return collection
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return None
    
    def delete_collection(self, collection_id: str) -> None:
        """
        Delete a collection and its associated data.
        
        Args:
            collection_id: Unique identifier of the collection to delete
            
        The method includes error handling and logging for failed deletions.
        """
        try:
            self.client.delete_collection(collection_id)
        except Exception as e:
            logger.error(f"Error deleting collection {collection_id}: {e}")