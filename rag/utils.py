"""
Utility Functions Module

This module provides logging functionality for the RAG chatbot system.
It sets up both file and console logging with appropriate formatting
and error handling.

Key Components:
    - Logger configuration
    - File and console handlers
    - Standardized log formatting
"""

import logging
import sys
from pathlib import Path

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Configure logging with file and console handlers.
    
    Args:
        log_dir: Directory to store log files (default: "logs")
        
    Returns:
        logging.Logger: Configured logger instance
        
    Sets up:
        - File logging with UTF-8 encoding
        - Console output for immediate feedback
        - Timestamp and level-based formatting
    """
    Path(log_dir).mkdir(exist_ok=True)
    
    logger = logging.getLogger('chatbot')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(
        Path(log_dir) / "chatbot.log",
        encoding='utf-8'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()