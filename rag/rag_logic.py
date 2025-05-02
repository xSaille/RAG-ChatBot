import gc
import ollama
import numpy as np

from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from rag.file_processor import PDFFileProcessor
from rag.settings import Settings
from rag.utils import logger

class RAGChatbot:
    def __init__(self, ollama_model_name: str = Settings.OLLAMA_MODEL_NAME, 
                 embedding_model: str = Settings.EMBEDDING_MODEL_NAME):
        """
        Initializes the RAG Chatbot using Ollama.
        Args:
            ollama_model_name: The name of the model to use in Ollama
            embedding_model: The SentenceTransformer model for embeddings
        """
        self.file_processor = PDFFileProcessor(model_name=embedding_model)
        self.ollama_model_name = ollama_model_name
        self.processed_data: List[Dict] = []
        self.processed_file_name: Optional[str] = None
        logger.info(f"RAGChatbot initialized. Using Ollama model: {self.ollama_model_name}")
        try:
            ollama.list()
            logger.info("Successfully connected to Ollama.")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama. Ensure Ollama server is running. Error: {e}")


    def process_uploaded_file(self, file_obj) -> str:
        """ Processes the uploaded PDF file. """
        if file_obj is None:
            return "No file uploaded. Please upload a PDF."
        print(f"Received file: {file_obj.name}")
        self.processed_file_name = file_obj.name
        self.processed_data = []
        gc.collect()
        
        self.processed_data = self.file_processor.process_document(file_obj.name)

        if not self.processed_data:
             self.processed_file_name = None
             return f"Failed to process {file_obj.name}. Check logs."
        num_chunks = len(self.processed_data)
        return f"Successfully processed '{self.processed_file_name}' into {num_chunks} chunks."

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """ Embeds the user query. """
        if not query: return None
        try:
            query_embedding = self.file_processor.model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )
            return query_embedding[0]
        except Exception as e:
            print(f"Error embedding query: {e}")
            return None

    def retrieve_relevant_chunks(self, query_embedding: np.ndarray, 
                               top_k: int = Settings.RETRIEVAL_TOP_K) -> List[str]:
        """ Retrieves top_k relevant chunk texts based on cosine similarity. """
        if query_embedding is None or not self.processed_data:
            logger.warning("No query embedding or processed data available")
            return []
        
        chunk_embeddings = np.array([chunk['embedding'] for chunk in self.processed_data 
                                   if chunk.get('embedding') is not None])
        if chunk_embeddings.shape[0] == 0:
            logger.warning("No chunk embeddings available for retrieval.")
            return []

        similarities = cosine_similarity(query_embedding.reshape(1, -1), chunk_embeddings)[0]
        k = min(top_k, len(similarities))
        if k == 0: return []
        
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        relevant_chunk_texts = [self.processed_data[i]['content'] for i in top_k_indices]
        logger.info(f"Retrieved {len(relevant_chunk_texts)} relevant chunks.")
        return relevant_chunk_texts

    def generate_answer_with_ollama(self, query: str, context_chunks: List[str], 
                                  history: List[Tuple[str, str]]) -> str:
        """ Generates answer using Ollama with context. """
        if not query: return "Please provide a query."

        try:
            with open('rag/prompt.txt', 'r') as f:
                system_message = f.read()
        except Exception as e:
            logger.error(f"Could not read prompt.txt: {e}")
            system_message = "You are a helpful AI assistant analyzing documents and answering questions."

        if context_chunks:
            context = "\n\n---\n\n".join(context_chunks)
            user_prompt_content = f"User Query: {query}\n\nPotentially Relevant Context Snippets:\n{context}"
        else:
            user_prompt_content = query
            logger.warning("No relevant chunks found for this query. Answering generally.")

        messages = [{"role": "system", "content": system_message}]
        limited_history = history[-(Settings.CHAT_HISTORY_LIMIT*2):] if history else []
        for turn in limited_history:
            if turn[0]: messages.append({"role": "user", "content": turn[0]})
            if turn[1] and len(messages) < (Settings.CHAT_HISTORY_LIMIT * 2 + 1):
                 messages.append({"role": "assistant", "content": turn[1]})
        
        messages.append({"role": "user", "content": user_prompt_content})

        logger.info(f"Calling Ollama ({self.ollama_model_name})")

        try:
            response = ollama.chat(
                model=self.ollama_model_name,
                messages=messages,
                options=Settings.GENERATION_CONFIG
            )
            answer = response['message']['content']
            logger.info(f"Received response from Ollama: {len(answer)} chars")
            return answer.strip()

        except Exception as e:
            logger.error(f"Error during Ollama API call: {e}")
            gc.collect()
            return f"Sorry, an error occurred while contacting the AI model: {e}"
        
    def chat_interface_logic(self, message: str, history: List[Tuple[str, str]], state: Dict) -> Tuple[str, List[Tuple[str, str]], Dict]:
        """ Handles the main RAG chat interaction flow. """
        
        if not self.processed_data and state.get('processed_data'):
             self.processed_data = state.get('processed_data', [])
             self.processed_file_name = state.get('processed_file_name')
             print(f"Restored state: {len(self.processed_data)} chunks for file {self.processed_file_name}")

        if not self.processed_data:
            history.append((message, "Please upload and process a PDF document first."))
            return "", history, state

        print(f"\nUser Query: {message}")

        retrieval_k = 5
        print(f"Retrieving top {retrieval_k} chunks.")

        query_embedding = self.embed_query(message)
        if query_embedding is None:
             history.append((message, "Sorry, I couldn't process your query embedding."))
             return "", history, state

        relevant_chunks = self.retrieve_relevant_chunks(query_embedding, top_k=retrieval_k)

        answer = self.generate_answer_with_ollama(message, relevant_chunks, history) 

        history.append((message, answer))

        state['processed_data'] = self.processed_data
        state['processed_file_name'] = self.processed_file_name

        return "", history, state 

    def get_initial_state(self) -> Dict:
        """ Returns the initial state for Gradio. """
        return {'processed_data': [], 'processed_file_name': None}
        
    def update_state_after_upload(self, status_message: str, state: Dict) -> Dict:
        """ Updates the Gradio state after file processing. """
        state['processed_data'] = self.processed_data
        state['processed_file_name'] = self.processed_file_name
        print(f"Updating state: {len(state.get('processed_data',[]))} chunks processed for file {state.get('processed_file_name')}")
        return state