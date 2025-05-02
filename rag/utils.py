import logging
import sys
from pathlib import Path

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging configuration"""
    Path(log_dir).mkdir(exist_ok=True)
    
    logger = logging.getLogger('chatbot')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(
        Path(log_dir) / "chatbot.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()