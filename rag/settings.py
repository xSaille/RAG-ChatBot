from typing import Dict, Any

class Settings:
    OLLAMA_MODEL_NAME: str = 'qwen2.5:3b'
    EMBEDDING_MODEL_NAME: str = 'BAAI/bge-m3'
    
    RETRIEVAL_TOP_K: int = 5
    EMBEDDING_INSTRUCTION: str = "Represent this document for retrieval: "
    CHAT_HISTORY_LIMIT: int = 5
    
    MIN_CHUNK_SIZE: int = 50
    CLUSTERING_DISTANCE_THRESHOLD: float = 0.6
    
    GENERATION_CONFIG: Dict[str, Any] = {
        "temperature": 0.5
    }
    
    GRADIO_SERVER_NAME: str = "localhost"
    GRADIO_SHARE: bool = False
    GRADIO_DEBUG: bool = False
    
    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get all settings as a dictionary"""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and isinstance(v, (str, int, float, dict, bool))}