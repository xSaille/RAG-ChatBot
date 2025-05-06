# RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that processes PDF documents and answers questions based on their content using Ollama and sentence-transformers. The chatbot uses a semantic search approach to find relevant information from processed documents and generates context-aware responses.

## Project Structure

```
ChatBot/
├── main.py                 # Main application entry point with Gradio UI
├── requirements.txt        # Python dependencies
├── rag/                   # Core RAG implementation
│   ├── db_manager.py      # ChromaDB management
│   ├── document_processor.py  # PDF processing and embedding
│   ├── document_store.py  # Document metadata storage
│   ├── prompt.txt         # System prompt for chat responses
│   ├── rag_logic.py      # Core chatbot logic
│   ├── settings.py       # Configuration settings
│   └── utils.py          # Utility functions and logging
└── logs/                 # Application logs
    └── chatbot.log      # Runtime logs
```

## Features

- **PDF Document Processing**: Convert PDF documents into searchable chunks with semantic embeddings
- **Semantic Search**: Find relevant information using BGE-M3 embeddings
- **Context-Aware Responses**: Generate responses using Ollama with relevant document context
- **Document Management**: Track and manage processed documents with metadata
- **Clean Web Interface**: User-friendly Gradio interface for document upload and chat
- **Persistent Storage**: Store document embeddings and metadata using ChromaDB

## Components

### 1. Document Processing (`document_processor.py`)
- Handles PDF file validation and text extraction
- Splits documents into semantic chunks
- Generates embeddings using BGE-M3 model
- Key functions:
  - `validate_pdf()`: Validates PDF files
  - `process_document()`: Processes PDFs into chunks with embeddings
  - `_generate_embedding()`: Creates embeddings for text chunks

### 2. Database Management (`db_manager.py`)
- Manages ChromaDB collections for vector storage
- Handles document persistence and retrieval
- Key functions:
  - `get_collection()`: Retrieves stored document collections
  - `create_collection()`: Creates new document collections
  - `delete_collection()`: Removes document collections

### 3. Document Store (`document_store.py`)
- Manages document metadata storage
- Tracks processed documents and their states
- Key functions:
  - `add_document()`: Stores document metadata
  - `remove_document()`: Removes document metadata
  - `get_document_info()`: Retrieves document information

### 4. RAG Logic (`rag_logic.py`)
- Implements core chatbot functionality
- Coordinates between components
- Handles chat flow and response generation
- Key functions:
  - `process_uploaded_file()`: Processes new documents
  - `retrieve_relevant_chunks()`: Finds relevant document sections
  - `chat_interface_logic()`: Manages chat interaction

## Installation

1. **Install Python Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Install and Start Ollama**:
- Follow instructions at https://ollama.ai/
- Pull the required model:
```bash
ollama pull qwen2.5:3b
```

## Configuration

Key settings in `settings.py`:

```python
# Model settings
OLLAMA_MODEL_NAME = 'qwen2.5:3b'

# Retrieval settings
RETRIEVAL_TOP_K = 5        # Number of chunks to retrieve
CHAT_HISTORY_LIMIT = 5     # Chat history context length

# Generation settings
GENERATION_CONFIG = {
    "temperature": 0.5,    # Response randomness
    "top_p": 0.9          # Nucleus sampling parameter
}

# Document processing
MIN_CHUNK_LENGTH = 10      # Minimum chunk size
SENTENCE_WINDOW_SIZE = 3   # Context window size
```

## Usage

1. **Start the Application**:
```bash
python main.py
```

2. **Upload a Document**:
- Click "Upload PDF" to select a PDF file
- Click "Process Document" to analyze the file
- Wait for processing confirmation

3. **Chat with the Document**:
- Type your questions in the chat input
- The system will:
  1. Find relevant sections from the document
  2. Generate context-aware responses
  3. Display the response in the chat interface

4. **Switch Documents**:
- Use the document dropdown to switch between processed documents
- Click "Load Document" to activate the selected document

## Error Handling

The system includes comprehensive error handling:
- PDF validation checks
- Embedding generation retries
- Document processing validation
- Database operation error recovery
- Chat interface state management

## Logging

Logs are stored in `logs/chatbot.log` and include:
- Document processing events
- Error messages
- Chat interactions
- System operations

## Development

### Adding New Features

1. Update relevant component files in the `rag/` directory
2. Add tests in `tests/test_rag_logic.py`
3. Update configuration in `settings.py` if needed
4. Document changes in comments and docstrings

### Running Tests

```bash
pytest tests/test_rag_logic.py
```

### Best Practices

- Follow existing error handling patterns
- Use type hints for function parameters
- Add docstrings for new functions
- Log significant operations and errors
- Validate document processing results

## Dependencies

- `gradio`: Web interface
- `ollama`: LLM integration
- `sentence-transformers`: Text embeddings
- `chromadb`: Vector storage
- `llama-index`: Document processing
- `PyMuPDF`: PDF handling
- Additional utilities in `requirements.txt`

## Limitations

- Currently supports PDF files only
- Memory usage scales with document size
- Response quality depends on Ollama model
- Processing time increases with document length

## Future Improvements

- Support for additional document types
- Improved chunk segmentation
- Memory optimization for large documents
- Enhanced error recovery
- Additional embedding models
- Response source attribution