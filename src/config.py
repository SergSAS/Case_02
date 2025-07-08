"""
Configuration module for text classification analysis.

This module handles all configuration settings including API keys,
model parameters, and file paths.
"""

import os
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv
import logging


class Config:
    """
    Configuration manager for the text classification project.
    
    Handles loading environment variables, API keys, and project settings.
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            env_file: Path to .env file. If None, looks for config/.env
        """
        self.project_root = Path(__file__).parent.parent
        if env_file:
            self.env_file = Path(env_file)
        else:
            self.env_file = self.project_root / "config" / ".env"
        self._load_environment()
        self._setup_directories()
        self._setup_logging()
    
    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        if self.env_file.exists():
            load_dotenv(self.env_file)
            logging.info(f"Loaded environment from {self.env_file}")
        else:
            logging.warning(f"Environment file not found: {self.env_file}")
    
    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results"
        self.logs_dir = self.project_root / "logs"
        
        for directory in [self.data_dir, self.results_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
    
    def _setup_logging(self) -> None:
        """Configure logging settings."""
        log_file = self.logs_dir / "analysis.log"
        
        # Remove existing handlers to avoid conflicts
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a'),
                logging.StreamHandler()
            ],
            force=True
        )
    
    @property
    def groq_api_key(self) -> Optional[str]:
        """Get Groq API key from environment."""
        return os.getenv("GROQ_API_KEY")
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY")
    
    @property
    def model_config(self) -> Dict[str, any]:
        """Get model configuration settings."""
        return {
            "traditional": {
                "vectorizer": {
                    "max_features": 15000,
                    "ngram_range": (1, 3),
                    "max_df": 0.9,
                    "min_df": 3,
                    "use_idf": True,
                    "sublinear_tf": True,
                    "stop_words": "english"
                },
                "classifier": {
                    "penalty": "l2",
                    "C": 1.0,
                    "solver": "liblinear",
                    "max_iter": 1000,
                    "random_state": 42
                }
            },
            "llm": {
                "model": os.getenv("LLM_MODEL", "gemma2-9b-it"),
                "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
                "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "10")),
                "rate_limit_delay": float(os.getenv("RATE_LIMIT_DELAY", "2.0"))
            }
        }
    
    @property
    def dataset_config(self) -> Dict[str, any]:
        """Get dataset configuration."""
        return {
            "name": "pietrolesci/eurlex-57k",
            "train_size": 5000,
            "validation_size": 1000,
            "test_size": 1000,
            "text_max_length": 1500,
            "classification_threshold": 2000  # Characters for binary classification
        }
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        Validate that required API keys are present.
        
        Returns:
            Dictionary with API key validation status
        """
        return {
            "groq": bool(self.groq_api_key),
            "openai": bool(self.openai_api_key)
        } 