"""
Configuration Settings Module
# ... (rest of your existing docstring)
"""

import os
import json
from typing import Dict, Any
from pathlib import Path

class Settings:
    """
    # ... (rest of your existing docstring)
    """

    GRADIO_SERVER_NAME: str = "127.0.0.1"
    GRADIO_SHARE: bool = False
    GRADIO_DEBUG: bool = True 
    
    OLLAMA_MODEL_NAME: str = 'qwen2.5:3b'
    
    RETRIEVAL_TOP_K: int = 5
    CHAT_HISTORY_LIMIT: int = 5
    
    GENERATION_CONFIG: Dict[str, Any] = {
        "temperature": 0.5,
        "top_p": 0.9
    }
    
    MIN_CHUNK_LENGTH: int = 10
    SENTENCE_WINDOW_SIZE: int = 3
    SUPPORTED_FILE_TYPES = ['.pdf']
    EMBEDDING_RETRIES: int = 3

    BASE_DIR = Path(__file__).resolve().parent.parent
    PROMPT_FILE_PATH = os.path.join(BASE_DIR, "rag", "prompt.txt")

    CHROMA_PERSIST_DIR = "./chroma_db"
    CHROMA_COLLECTION_PREFIX = "doc_"
    METADATA_DIR = os.path.join(CHROMA_PERSIST_DIR, "metadata")
    PROCESSED_FILES_PATH = os.path.join(METADATA_DIR, "processed_files.json")
    DOCUMENT_METADATA_PATH = os.path.join(METADATA_DIR, "document_metadata.json")
    
    @classmethod
    def initialize_directories(cls) -> None:
        """
        # ... (rest of your existing method)
        """
        try:
            chroma_persist_path = Path(cls.CHROMA_PERSIST_DIR)
            chroma_persist_path.mkdir(parents=True, exist_ok=True)
            
            metadata_path = chroma_persist_path / "metadata"
            metadata_path.mkdir(parents=True, exist_ok=True)
            
            cls.PROCESSED_FILES_PATH = str(metadata_path / "processed_files.json")
            cls.DOCUMENT_METADATA_PATH = str(metadata_path / "document_metadata.json")

            cls._initialize_json_file(cls.PROCESSED_FILES_PATH, {})
            cls._initialize_json_file(cls.DOCUMENT_METADATA_PATH, {})
            
        except Exception as e:
            print(f"Failed to initialize directories: {e}")
            raise
    
    @staticmethod
    def _initialize_json_file(filepath: str, default_content: Dict) -> None:
        """
        # ... (rest of your existing method)
        """
        path_obj = Path(filepath)
        if not path_obj.exists():
            try:
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(default_content, f, indent=2)
            except Exception as e:
                print(f"Failed to initialize JSON file at {filepath}: {e}")
    
    @classmethod
    def load_processed_files(cls) -> Dict[str, str]:
        """
        Load mapping of processed files to collection IDs.
        
        Returns:
            Dict[str, str]: Mapping of filenames to collection IDs
            
        Creates empty mapping if file doesn't exist or is corrupted.
        """
        cls.initialize_directories()
        try:
            with open(cls.PROCESSED_FILES_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Processed files JSON ({cls.PROCESSED_FILES_PATH}) not found or corrupted ({e}). Initializing with empty data.")
            cls._initialize_json_file(cls.PROCESSED_FILES_PATH, {})
            return {}
        except Exception as e:
            print(f"Unexpected error loading processed files JSON ({cls.PROCESSED_FILES_PATH}): {e}. Initializing with empty data.")
            cls._initialize_json_file(cls.PROCESSED_FILES_PATH, {})
            return {}

    @classmethod
    def load_document_metadata(cls) -> Dict[str, Dict[str, Any]]:
        """
        Load document metadata.
        
        Returns:
            Dict[str, Dict[str, Any]]: Document metadata indexed by filename
            
        Creates empty metadata if file doesn't exist or is corrupted.
        """
        cls.initialize_directories()
        try:
            with open(cls.DOCUMENT_METADATA_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Document metadata JSON ({cls.DOCUMENT_METADATA_PATH}) not found or corrupted ({e}). Initializing with empty data.")
            cls._initialize_json_file(cls.DOCUMENT_METADATA_PATH, {})
            return {}
        except Exception as e:
            print(f"Unexpected error loading document metadata JSON ({cls.DOCUMENT_METADATA_PATH}): {e}. Initializing with empty data.")
            cls._initialize_json_file(cls.DOCUMENT_METADATA_PATH, {})
            return {}