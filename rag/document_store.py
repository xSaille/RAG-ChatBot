"""
Document Storage Module

This module manages document metadata persistence and retrieval. It provides
a simple interface for storing and accessing information about processed
documents, including their collection IDs and processing statistics.

Key Components:
    - Document metadata management
    - JSON-based persistence
    - Basic error handling
"""

import json
from datetime import datetime
from typing import Dict, Optional, Any
from rag.settings import Settings as AppSettings
from rag.utils import logger

class DocumentStore:
    """
    Handles document metadata storage and retrieval.
    
    This class manages:
    1. Document metadata persistence
    2. Processing statistics
    3. Collection mapping
    
    Attributes:
        metadata: Dictionary containing document metadata and stats
    """
    
    def __init__(self):
        """
        Initialize document store and load existing metadata.
        
        Loads existing metadata from disk or creates new if none exists.
        """
        self.metadata = AppSettings.load_document_metadata()
        self._save_metadata()
    
    def _save_metadata(self) -> None:
        """
        Save metadata to disk.
        
        Persists the current state of document metadata to JSON storage,
        with basic error handling and logging.
        """
        try:
            with open(AppSettings.DOCUMENT_METADATA_PATH, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def add_document(self, filename: str, collection_id: str, stats: Dict[str, Any]) -> None:
        """
        Add or update document metadata.
        
        Args:
            filename: Name of the processed document
            collection_id: ID of the associated ChromaDB collection
            stats: Processing statistics and metadata
            
        The method stores:
            - Collection ID for vector storage lookup
            - Processing timestamp
            - Document statistics (chunks, pages, etc.)
        """
        self.metadata[filename] = {
            'collection_id': collection_id,
            'processed_date': datetime.now().isoformat(),
            'stats': stats
        }
        self._save_metadata()
    
    def remove_document(self, filename: str) -> None:
        """
        Remove document metadata.
        
        Args:
            filename: Name of the document to remove
            
        Deletes all stored metadata for the specified document.
        """
        if filename in self.metadata:
            del self.metadata[filename]
            self._save_metadata()
    
    def get_document_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata.
        
        Args:
            filename: Name of the document to retrieve info for
            
        Returns:
            Dict or None: Document metadata if found, None if not found
            
        The returned metadata includes:
            - Collection ID
            - Processing timestamp
            - Document statistics
        """
        return self.metadata.get(filename)