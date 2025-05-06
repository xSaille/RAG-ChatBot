"""
RAG (Retrieval-Augmented Generation) Logic Module

This module implements the core chatbot functionality, coordinating between
document processing, vector storage, and response generation. It serves as
the main orchestrator for the RAG system.

Key Components:
    - Document processing coordination
    - Query embedding and retrieval
    - Chat interface logic
    - Response generation with context
"""

import json
import ollama
import numpy as np # Ensure NumPy is imported
from typing import List, Dict, Optional, Tuple, Any
from llama_index.core import Settings as LlamaIndexSettings 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rag.settings import Settings as AppSettings
from rag.utils import logger
from rag.document_store import DocumentStore
from rag.db_manager import DBManager
from rag.document_processor import DocumentProcessor

class RAGChatbot:
    """
    Main RAG chatbot implementation.
    
    This class coordinates:
    1. Document processing and storage
    2. Query processing and retrieval
    3. Chat interface management
    4. Response generation
    
    Attributes:
        embed_model: BGE-M3 embedding model
        ollama_model_name: Name of the Ollama model for responses
        doc_store: Document metadata manager
        db_manager: ChromaDB interface
        doc_processor: Document processing component
        processed_data: Currently loaded document chunks
        processed_file_name: Name of the active document
        processed_files: Map of available documents
    """
    
    def __init__(self, ollama_model_name: str = AppSettings.OLLAMA_MODEL_NAME):
        """
        Initialize the RAG chatbot system.
        
        Args:
            ollama_model_name: Name of the Ollama model to use (default: from settings)
        """
        try:
            self.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-m3", 
                embed_batch_size=1, 
                trust_remote_code=True
            )
            LlamaIndexSettings.embed_model = self.embed_model
            logger.info("Successfully initialized embedding model")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}", exc_info=True)
            raise
        
        self.ollama_model_name = ollama_model_name
        self.doc_store = DocumentStore()
        self.db_manager = DBManager()
        self.doc_processor = DocumentProcessor(self.embed_model)
        
        self.processed_data = []
        self.processed_file_name = None
        self.processed_files = self.db_manager.processed_files

    def get_available_documents(self) -> List[str]:
        """
        Get list of previously processed documents.
        """
        return list(self.processed_files.keys())

    def load_document(self, filename: str) -> str:
        """
        Load a previously processed document.
        """
        if not filename or filename == "Select a document..." or filename is None: # Added None check
            logger.warning("Load document called with no or placeholder filename.")
            return "Please select a valid document"
            
        if filename not in self.processed_files:
            logger.warning(f"Attempted to load unprocessed document: {filename}")
            return f"Document '{filename}' has not been processed yet"

        collection_id = self.processed_files.get(filename)
        if not collection_id:
            logger.error(f"Collection ID not found for document: {filename} in processed_files map.")
            return f"Internal error: Collection ID missing for '{filename}'. Please reprocess."

        logger.info(f"Loading document '{filename}' with collection ID '{collection_id}'")
        collection = self.db_manager.get_collection(collection_id)

        if not collection:
            logger.error(f"Could not retrieve collection '{collection_id}' for document '{filename}' from DBManager.")
            return f"Document data not found for '{filename}'. It might have been deleted or an error occurred. Please process the document again."
        
        logger.debug(f"Fetching data with include: ['documents', 'embeddings', 'metadatas'] for '{filename}' (Collection ID: {collection_id})")
        
        try:
            items = collection.get(include=['documents', 'embeddings', 'metadatas']) 
            
            if not items:
                logger.error(f"Collection.get() returned empty items for {filename} (collection: {collection_id})")
                return f"Document data is empty for '{filename}'. Please process the document again."

            if 'ids' not in items or not items['ids']: 
                 logger.warning(f"No items (IDs) found in collection {collection_id} for document {filename}. Returned items: {str(items)[:500]}...")
                 self.processed_data = [] 
                 self.processed_file_name = filename 
                 return f"Warning: No data chunks found in '{filename}'. The document might be empty or processing yielded no chunks."

            num_ids = len(items['ids'])
            logger.info(f"Retrieved {num_ids} item IDs from collection {collection_id} for {filename}.")
            logger.debug(f"Raw collection data for '{filename}' (first 500 chars): {str(items)[:500]}")

            if 'embeddings' not in items or items['embeddings'] is None:
                logger.error(f"Invalid or missing 'embeddings' key or None value in collection items for {filename} (collection: {collection_id}).")
                logger.debug(f"Full items dictionary for {filename} (if embeddings missing): {items}")
                return f"Document data for '{filename}' is corrupted (missing embeddings). Please process the document again."
            
            if not all(key in items for key in ['documents', 'metadatas']): 
                missing_keys = [key for key in ['documents', 'metadatas'] if key not in items]
                logger.error(f"Missing required data fields {missing_keys} in collection for {filename} (collection: {collection_id})")
                return f"Document data for '{filename}' is incomplete. Please process the document again."
                
            try:
                self.processed_data = []
                
                if not (len(items.get('documents', [])) == num_ids and \
                        len(items.get('embeddings', [])) == num_ids and \
                        len(items.get('metadatas', [])) == num_ids):
                    logger.error(f"Data inconsistency in collection {collection_id} for {filename}: "
                                 f"IDs ({num_ids}), "
                                 f"Docs ({len(items.get('documents', []))}), " 
                                 f"Embeds ({len(items.get('embeddings', []))}), "
                                 f"Metas ({len(items.get('metadatas', []))}).")
                    return f"Data inconsistency found in '{filename}'. Please reprocess."

                for i in range(num_ids):
                    current_embedding_original = items['embeddings'][i]
                    
                    if current_embedding_original is None:
                        logger.warning(f"Individual embedding for item ID {items['ids'][i]} (index {i}) in document {filename} is None. Skipping this item.")
                        continue 
                    
                    processed_embedding = None
                    if isinstance(current_embedding_original, list):
                        processed_embedding = current_embedding_original
                    elif isinstance(current_embedding_original, np.ndarray):
                        logger.debug(f"Embedding for item ID {items['ids'][i]} (index {i}) is numpy.ndarray, converting to list.")
                        processed_embedding = current_embedding_original.tolist()
                    else:
                        logger.warning(f"Individual embedding for item ID {items['ids'][i]} (index {i}) in document {filename} is not a list or numpy.ndarray (type: {type(current_embedding_original)}). Skipping this item.")
                        continue

                    current_metadata_item = items['metadatas'][i]
                    if not isinstance(current_metadata_item, dict) or 'metadata' not in current_metadata_item:
                        logger.warning(f"Metadata for item ID {items['ids'][i]} (index {i}) in document {filename} is not a dict or missing 'metadata' key. Item: {current_metadata_item}. Skipping this item.")
                        continue
                    
                    try:
                        parsed_metadata = json.loads(current_metadata_item['metadata'])
                    except json.JSONDecodeError as je:
                        logger.warning(f"JSONDecodeError for metadata of item ID {items['ids'][i]} (index {i}) in document {filename}. Error: {je}. Metadata string: '{current_metadata_item['metadata']}'. Skipping this item.")
                        continue

                    self.processed_data.append({
                        'content': items['documents'][i],
                        'embedding': processed_embedding,
                        'metadata': parsed_metadata 
                    })
                
                if not self.processed_data and num_ids > 0 : 
                    logger.error(f"All {num_ids} items for document {filename} had corrupt embeddings or other processing issues after type conversion. No data loaded.")
                    return f"All data chunks for '{filename}' have corrupt embeddings or processing issues. Please reprocess the document."

            except (IndexError, KeyError, TypeError, json.JSONDecodeError) as e: 
                logger.error(f"Error parsing or structuring collection data for {filename} (collection: {collection_id}): {e}", exc_info=True)
                return f"Error parsing document data for '{filename}'. Please process the document again."
            
            self.processed_file_name = filename
            logger.info(f"Successfully loaded {len(self.processed_data)} chunks from {filename}")
            return f"Loaded {len(self.processed_data)} chunks from {filename}"
            
        except ValueError as ve: 
            logger.error(f"ValueError during collection.get for {filename} (collection: {collection_id}): {ve}", exc_info=True)
            return f"Error retrieving data for '{filename}' due to invalid parameters. {ve}"
        except Exception as e:
            logger.error(f"Unexpected error loading document {filename} (collection: {collection_id}): {e}", exc_info=True)
            return f"Error loading document '{filename}'. Please try again."

    def process_uploaded_file(self, file_obj) -> str:
        """
        Process an uploaded PDF file.
        """
        if not file_obj or not hasattr(file_obj, 'name') or not file_obj.name.lower().endswith('.pdf'):
            logger.warning("Invalid file uploaded or not a PDF.")
            return "Please upload a valid PDF file."
        
        original_filename_for_logging = getattr(file_obj, 'orig_name', file_obj.name)
        logger.info(f"Starting processing for uploaded file: {original_filename_for_logging}")

        try:
            filename, collection_id, embeddings_data = self.doc_processor.process_document(file_obj) 
            
            if not filename or not collection_id:
                logger.error(f"Document processing failed for {original_filename_for_logging}. Received: filename={filename}, collection_id={collection_id}, has_embeddings_data={bool(embeddings_data)}")
                return "Error processing document. Document processor returned insufficient data (filename/collection_id)."
            
            if embeddings_data is None:
                logger.error(f"Document processing for {filename} returned None for embeddings_data.")
                return "Error processing document: No embedding data returned."

            logger.info(f"Document processor returned {len(embeddings_data)} items for {filename}.")
            logger.debug(f"Raw embeddings_data from doc_processor for {filename} (first 500 chars): {str(embeddings_data)[:500]}...")

            if not isinstance(embeddings_data, list) or not all(isinstance(item, dict) for item in embeddings_data):
                logger.error(f"Embeddings data for {filename} is not a list of dictionaries as expected.")
                return "Error: Document processing failed - invalid data structure from processor."

            embeddings_for_chroma = []
            valid_embeddings_data_for_state = []
            documents_for_chroma = []
            metadatas_for_chroma = []
            ids_for_chroma = []

            for i, item in enumerate(embeddings_data):
                if 'embedding' not in item or item['embedding'] is None:
                    logger.warning(f"Item {i} for {filename} has missing or None embedding. Skipping. Content: {item.get('content', 'N/A')[:50]}...")
                    continue
                if not isinstance(item['embedding'], list) or not all(isinstance(x, (float, int)) for x in item['embedding']):
                    logger.warning(f"Item {i} for {filename} has an invalid embedding format. Skipping. Type: {type(item['embedding'])}. Content: {item.get('content', 'N/A')[:50]}...")
                    continue
                if 'content' not in item or 'metadata' not in item:
                    logger.warning(f"Item {i} for {filename} is missing 'content' or 'metadata'. Skipping. Item: {item}")
                    continue

                embeddings_for_chroma.append(item['embedding'])
                documents_for_chroma.append(item['content'])
                metadatas_for_chroma.append({'metadata': json.dumps(item['metadata'])})
                ids_for_chroma.append(str(i)) 
                valid_embeddings_data_for_state.append(item)
            
            if not valid_embeddings_data_for_state:
                logger.warning(f"No valid embeddings/chunks found for {filename} after filtering. Original: {len(embeddings_data)}.")
                self.processed_file_name = filename
                self.processed_data = []
                self.processed_files[filename] = collection_id
                self.db_manager._save_processed_files()
                self.doc_store.add_document(filename, collection_id, {'num_chunks': 0, 'original_num_chunks_processed': len(embeddings_data)})
                return f"Successfully processed {filename}: 0 valid chunks found."

            logger.info(f"Proceeding with {len(valid_embeddings_data_for_state)} valid chunks for {filename}.")

            if filename in self.processed_files:
                old_collection_id = self.processed_files[filename]
                if old_collection_id != collection_id:
                     logger.info(f"Content for {filename} changed (new ID {collection_id}, old {old_collection_id}). Deleting old collection.")
                     self.db_manager.delete_collection(old_collection_id)
                else:
                    logger.info(f"Re-processing {filename} (ID {collection_id}). Clearing old data.")
                    self.db_manager.delete_collection(collection_id)
            
            self.processed_file_name = filename
            self.processed_data = valid_embeddings_data_for_state
            
            logger.info(f"Creating/getting collection '{collection_id}' for {filename}")
            collection = self.db_manager.create_collection(
                collection_id,
                metadata={"source": filename}
            )
            if not collection:
                 logger.error(f"Failed to create or get collection '{collection_id}' for {filename}")
                 return f"Error: Could not create database collection for {filename}."
            
            logger.info(f"Adding {len(ids_for_chroma)} chunks to collection '{collection_id}' for {filename}.")
            
            collection.add(
                ids=ids_for_chroma,
                documents=documents_for_chroma,
                embeddings=embeddings_for_chroma, 
                metadatas=metadatas_for_chroma
            )
            logger.info(f"Successfully added {len(ids_for_chroma)} items to collection '{collection_id}' for {filename}.")
            
            self.processed_files[filename] = collection_id
            self.db_manager._save_processed_files() 
            
            self.doc_store.add_document(filename, collection_id, {
                'num_chunks': len(valid_embeddings_data_for_state),
                'original_num_chunks_processed': len(embeddings_data)
            })
            
            return f"Successfully processed {len(valid_embeddings_data_for_state)} chunks from {filename}"

        except Exception as e:
            logger.error(f"Error processing file {original_filename_for_logging}: {e}", exc_info=True)
            self.processed_data = []
            self.processed_file_name = None
            return f"Error processing file '{original_filename_for_logging}': {str(e)}"

    def retrieve_relevant_chunks(self, query_embedding: np.ndarray) -> List[str]:
        """
        Retrieve relevant chunks for a query.
        """
        if query_embedding is None :
            logger.warning("Query embedding is None, cannot retrieve chunks.")
            return []
        if not self.processed_file_name:
            logger.warning("No document processed or loaded, cannot retrieve chunks.")
            return []
            
        try:
            collection_id = self.processed_files.get(self.processed_file_name)
            if not collection_id:
                logger.error(f"Cannot retrieve chunks. No collection ID for file: {self.processed_file_name}")
                return []

            collection = self.db_manager.get_collection(collection_id)
            if not collection:
                logger.error(f"Cannot retrieve chunks. Collection {collection_id} not found for file: {self.processed_file_name}")
                return []

            logger.info(f"Querying collection {collection_id} for top {AppSettings.RETRIEVAL_TOP_K} results.")
            query_embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            results = collection.query(
                query_embeddings=[query_embedding_list], 
                n_results=AppSettings.RETRIEVAL_TOP_K,
                include=['documents'] 
            )
            
            retrieved_docs = results['documents'][0] if results and 'documents' in results and results['documents'] else []
            logger.info(f"Retrieved {len(retrieved_docs)} chunks for the query.")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error retrieving chunks for file {self.processed_file_name}: {e}", exc_info=True)
            return []

    def chat_interface_logic(self, message: str, history_from_gradio: List[Tuple[Optional[str], Optional[str]]], 
                           state: Dict) -> Tuple[str, List[Tuple[Optional[str], Optional[str]]], Dict]:
        """
        Handle chat interface logic and response generation.
        
        Args:
            message: User's input message
            history_from_gradio: Previous chat interactions from Gradio (List[Tuple[str, str]])
            state: Current chat interface state (Dict)
            
        Returns:
            tuple:
                - str: Empty string (Gradio requirement for some components)
                - List[Tuple[str, str]]: Updated chat history for Gradio Chatbot component
                - Dict: Updated interface state
        """
        logger.info(f"Received message: '{message}'")
        logger.debug(f"Input history_from_gradio (expecting list of tuples): {history_from_gradio}")

        if not self.processed_data and state.get('processed_data'):
             logger.info("Restoring RAGChatbot's processed_data and processed_file_name from Gradio state.")
             self.processed_data = state.get('processed_data', [])
             self.processed_file_name = state.get('processed_file_name')
        elif self.processed_file_name != state.get('processed_file_name') and state.get('processed_file_name'):
            logger.info(f"Gradio state file ('{state.get('processed_file_name')}') differs from RAG ('{self.processed_file_name}'). Syncing.")
            self.processed_data = state.get('processed_data', [])
            self.processed_file_name = state.get('processed_file_name')
        
        current_gradio_history: List[Tuple[Optional[str], Optional[str]]] = list(history_from_gradio) if history_from_gradio else []

        if not self.processed_file_name or not self.processed_data:
            logger.warning("No document loaded or no processed data available. Prompting user.")
            updated_gradio_history = current_gradio_history + [(message, "Please upload and process a PDF document first, or select an already processed one.")]
            return "", updated_gradio_history, state

        logger.info(f"Generating embedding for query: '{message}' using loaded document '{self.processed_file_name}'")
        query_embedding = self.embed_model.get_text_embedding(message)

        if query_embedding is None:
             logger.error("Failed to generate query embedding.")
             updated_gradio_history = current_gradio_history + [(message, "Sorry, I couldn't process your query (embedding generation failed).")]
             return "", updated_gradio_history, state
        logger.info("Query embedding generated successfully.")

        relevant_chunks = self.retrieve_relevant_chunks(query_embedding)
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks.")
        if not relevant_chunks:
            logger.warning("No relevant chunks found for the query.")

        try:
            with open(AppSettings.PROMPT_FILE_PATH, 'r', encoding='utf-8') as f: 
                system_message = f.read()
            logger.debug("Loaded system prompt successfully.")
        except FileNotFoundError:
            logger.error(f"Prompt file not found at {AppSettings.PROMPT_FILE_PATH}. Using default system message.")
            system_message = "You are a helpful AI assistant analyzing documents."
        except Exception as e:
            logger.error(f"Error reading prompt file: {e}. Using default system message.", exc_info=True)
            system_message = "You are a helpful AI assistant analyzing documents."

        context = "\n\n---\n\n".join(relevant_chunks) if relevant_chunks else "No specific context found for this query in the document."
        user_prompt_for_llm = f"User Query: {message}\n\nPotentially Relevant Context Snippets:\n{context}"
        logger.debug(f"User prompt for Ollama (first 500 chars): {user_prompt_for_llm[:500]}...")

        ollama_messages = [{"role": "system", "content": system_message}]
        history_limit_for_ollama_turns = AppSettings.CHAT_HISTORY_LIMIT
        
        for user_msg, assistant_msg in current_gradio_history[-history_limit_for_ollama_turns:]:
            if user_msg is not None:
                ollama_messages.append({"role": "user", "content": str(user_msg)})
            if assistant_msg is not None:
                ollama_messages.append({"role": "assistant", "content": str(assistant_msg)})
        
        ollama_messages.append({"role": "user", "content": user_prompt_for_llm})
        logger.debug(f"Final messages for Ollama: {ollama_messages}")

        try:
            logger.info(f"Sending request to Ollama model: {self.ollama_model_name}")
            response = ollama.chat(
                model=self.ollama_model_name,
                messages=ollama_messages,
                options=AppSettings.GENERATION_CONFIG
            )
            answer = response['message']['content'].strip()
            logger.info(f"Received answer from Ollama (first 100 chars): {answer[:100]}...")
        except Exception as e:
            logger.error(f"Error generating answer with Ollama: {e}", exc_info=True)
            answer = "I encountered an error while trying to generate a response. Please try again."

        updated_gradio_history = current_gradio_history + [(message, answer)]
        
        state['processed_data'] = self.processed_data 
        state['processed_file_name'] = self.processed_file_name
        logger.debug(f"Updated Gradio state: processed_file_name='{self.processed_file_name}', processed_data_len={len(self.processed_data if self.processed_data else [])}")
        logger.debug(f"Returning history for Gradio (list of tuples): {updated_gradio_history}")

        return "", updated_gradio_history, state 

    def get_initial_state(self) -> Dict:
        """
        Get initial state for the chat interface.
        """
        logger.debug("Getting initial Gradio state for RAGChatbot.")
        return {'processed_data': [], 'processed_file_name': None}

    def update_state_after_upload_or_load(self, status: str, state: Dict) -> Dict:
        """
        Update Gradio interface state after document upload or load.
        """
        logger.info(f"Attempting to update Gradio state. Status: '{status}'")
        if self.processed_file_name and ("Successfully processed" in status or "Loaded" in status and "chunks from" in status):
            updated_state = {
                'processed_data': self.processed_data,
                'processed_file_name': self.processed_file_name
            }
            logger.info(f"Gradio state updated: file='{self.processed_file_name}', data_len={len(self.processed_data if self.processed_data else [])}")
            return updated_state
        
        logger.info("Gradio state not updated based on RAGChatbot state or status message.")
        return state