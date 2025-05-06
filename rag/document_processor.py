"""
Document Processing Module

This module handles PDF document processing, including validation, text extraction,
chunking, and embedding generation. It serves as the primary interface for
converting PDF documents into searchable, embedded text chunks.

Key Components:
    - PDF validation and handling
    - Text extraction and chunking
    - Embedding generation using BGE-M3 model
    - Temporary file management

The module ensures documents are properly validated before processing and handles
the conversion of document content into a format suitable for semantic search.
"""

import os
import hashlib
from typing import List, Dict, Any, Optional
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceWindowNodeParser
from rag.settings import Settings as AppSettings
from rag.utils import logger

class DocumentProcessor:
    """
    Handles document processing and embedding generation.
    
    This class is responsible for:
    1. Validating PDF documents
    2. Extracting text content
    3. Splitting content into semantic chunks
    4. Generating embeddings for chunks
    
    Attributes:
        embed_model: The embedding model used for generating vector representations
        node_parser: Parser for splitting documents into semantic chunks
    """
    
    def __init__(self, embed_model):
        """
        Initialize the document processor.
        
        Args:
            embed_model: Model instance for generating text embeddings
                       (typically HuggingFace BGE-M3)
        """
        self.embed_model = embed_model
        self.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=AppSettings.SENTENCE_WINDOW_SIZE,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )
    
    def validate_pdf(self, file_path: str) -> tuple[bool, int]:
        """
        Validate a PDF file and count its pages.
        
        Args:
            file_path: Path to the PDF file to validate
            
        Returns:
            tuple:
                - bool: True if valid PDF, False otherwise
                - int: Number of pages (0 if invalid)
        """

        try:
            import fitz
        except ImportError:
            logger.error("PyMuPDF (fitz) is not installed. Please install it to validate PDFs.")
            return False, 0

        try:
            with fitz.open(file_path) as pdf:
                if not pdf.is_pdf or len(pdf) == 0:
                    logger.warning(f"File '{file_path}' is not a valid PDF or is empty.")
                    return False, 0
                logger.info(f"PDF '{file_path}' validated successfully with {len(pdf)} pages.")
                return True, len(pdf)
        except Exception as e:
            logger.error(f"Error validating PDF '{file_path}': {e}", exc_info=True)
            return False, 0
    
    def process_document(self, file_obj) -> tuple[Optional[str], Optional[str], Optional[List[Dict[str, Any]]]]:
        """
        Process a PDF document and generate embeddings.
        
        This method handles the complete document processing pipeline:
        1. File validation
        2. Text extraction
        3. Chunking
        4. Embedding generation
        
        Args:
            file_obj: File object representing the PDF document (e.g., from Gradio upload)
            
        Returns:
            tuple:
                - str or None: Filename if successful, None if failed
                - str or None: Collection ID if successful, None if failed
                - List[Dict] or None: List of processed chunks with embeddings if successful,
                                    None if failed
                
        Each chunk in the returned list contains:
            - content: The text content
            - embedding: Vector representation (list of floats)
            - metadata: Additional chunk information from LlamaIndex
        """
        if not file_obj or not hasattr(file_obj, 'name') or not file_obj.name.lower().endswith('.pdf'):
            logger.error("Invalid file object provided to process_document. Not a PDF or missing name.")
            return None, None, None
        
        original_filename = getattr(file_obj, 'orig_name', os.path.basename(file_obj.name))
        filename_for_processing = os.path.basename(file_obj.name)
        
        logger.info(f"Starting document processing for '{original_filename}' (processing as '{filename_for_processing}')")
        
        temp_file_path = file_obj.name

        try:
            is_valid, page_count = self.validate_pdf(temp_file_path)
            if not is_valid:
                logger.error(f"Invalid PDF file: {original_filename}")
                return None, None, None

            logger.info(f"Validated PDF '{original_filename}' with {page_count} pages.")

            with open(temp_file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            collection_id = f"{AppSettings.CHROMA_COLLECTION_PREFIX}{file_hash}"
            logger.info(f"Generated collection ID for '{original_filename}': {collection_id}")
            
            reader = SimpleDirectoryReader(input_files=[temp_file_path])
            documents = reader.load_data()
            
            if not documents:
                logger.error(f"No content extracted from document: {original_filename}")
                return None, None, None
            
            logger.info(f"Successfully extracted {len(documents)} LlamaIndex Document objects from '{original_filename}'.")
            
            nodes = self.node_parser.get_nodes_from_documents(documents)
            if not nodes:
                logger.warning(f"No text nodes/chunks generated from LlamaIndex documents for '{original_filename}'. The document might be empty or content unsuitable for chunking.")
                return original_filename, collection_id, []

            logger.info(f"Generated {len(nodes)} text chunks/nodes for '{original_filename}'.")
            
            embeddings_data = []
            failed_chunks_embedding = 0
            
            for i, node in enumerate(nodes):
                content = node.get_content(metadata_mode="all").strip()
                
                if len(content) < AppSettings.MIN_CHUNK_LENGTH:
                    logger.debug(f"Skipping chunk {i} from '{original_filename}': too short ({len(content)} chars). Content: '{content[:100]}...'")
                    continue
                
                logger.debug(f"Generating embedding for chunk {i} from '{original_filename}' ({len(content)} chars).")
                embedding = self._generate_embedding(content)
                
                if embedding is None:
                    failed_chunks_embedding += 1
                    logger.error(f"Failed to generate embedding for chunk {i} from '{original_filename}'. Content: '{content[:100]}...'")
                    continue
                
                embeddings_data.append({
                    'content': node.get_content(),
                    'embedding': embedding,
                    'metadata': node.metadata
                })
                logger.debug(f"Successfully generated embedding for chunk {i} from '{original_filename}'.")


            if failed_chunks_embedding > 0:
                logger.warning(f"Failed to generate embeddings for {failed_chunks_embedding} out of {len(nodes)} chunks from '{original_filename}'.")
                
            if not embeddings_data:
                logger.error(f"No valid embeddings generated for any chunks in '{original_filename}'.")
                return original_filename, collection_id, [] 
            
            logger.info(f"Successfully generated {len(embeddings_data)} embeddings for '{original_filename}'.")
            return original_filename, collection_id, embeddings_data
            
        except Exception as e:
            logger.error(f"Error processing document '{original_filename}': {e}", exc_info=True)
            return None, None, None
    
    def _generate_embedding(self, content: str) -> Optional[List[float]]:
        """
        Generate embedding vector for a text chunk.
        
        Args:
            content: Text content to generate embedding for
            
        Returns:
            List[float] or None: Embedding vector if successful, None if failed
        """
        if not content or not content.strip():
            logger.error("Empty content provided for embedding generation.")
            return None
            
        try:
            max_retries = AppSettings.EMBEDDING_RETRIES
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Attempt {attempt + 1}/{max_retries} to generate embedding for content (len: {len(content)}): '{content[:100]}...'")
                    embedding = self.embed_model.get_text_embedding(content)

                    if embedding is None:
                        logger.warning(f"Embedding model returned None on attempt {attempt + 1} for content.")
                        raise ValueError("Embedding model returned None")
                        
                    if hasattr(embedding, 'tolist'):
                        embedding = embedding.tolist()
                        
                    if not isinstance(embedding, list) or not embedding:
                        logger.warning(f"Embedding is not a list or is empty on attempt {attempt + 1}.")
                        raise ValueError("Invalid embedding format: not a list or empty")
                    if not all(isinstance(x, (int, float)) for x in embedding):
                        logger.warning(f"Embedding contains non-numeric values on attempt {attempt + 1}.")
                        raise ValueError("Invalid embedding values: contains non-numeric")
                        
                    logger.debug(f"Successfully generated embedding on attempt {attempt + 1}.")
                    return embedding
                    
                except Exception as e:
                    logger.warning(f"Embedding generation attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        logger.error(f"All {max_retries} attempts to generate embedding failed for content: '{content[:100]}...'")
                        raise
        except Exception as e:
            logger.error(f"Critical error generating embedding after {max_retries} attempts: {e}", exc_info=True)
            return None