# RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that can process PDF documents and answer questions based on their content using Ollama and sentence-transformers.

## Project Structure

```
ChatBot/
├── rag/
│   ├── file_processor.py    # PDF processing and text chunking
│   ├── rag_logic.py        # Core RAG implementation
│   ├── settings.py         # Configuration settings
│   ├── prompt.txt         # System prompt for the chatbot
│   └── utils.py           # Utility functions (logging, etc.)
├── tests/
│   └── test_rag_logic.py   # Unit tests
├── main.py                 # Gradio web interface
└── requirements.txt        # Project dependencies
```

## Features

- PDF document processing with intelligent text chunking
- Semantic search using sentence-transformers
- Chat interface with conversation history
- Configurable settings
- Proper logging system
- Unit tests

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install and start Ollama server:
- Follow instructions at https://ollama.ai/
- Pull the required model:
```bash
ollama pull qwen2.5:3b
```

3. Run the application:
```bash
python main.py
```

## Configuration

Key settings can be modified in `rag/settings.py`:
- Model settings (Ollama model, embedding model)
- RAG parameters (top-k retrieval, chunk size, etc.)
- Generation settings (temperature, etc.)
- Server settings

## Development

- Run tests: `python -m pytest tests/`
- Logs are stored in the `logs/` directory
- See `rag/settings.py` for configurable parameters

## Requirements

- Python 3.8+
- Ollama
- GPU recommended but not required