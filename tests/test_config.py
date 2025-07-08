"""Tests for configuration module."""

import os
import tempfile
import pytest
from pathlib import Path
from src.config import Config


class TestConfig:
    """Test cases for Config class."""
    
    def test_config_initialization(self):
        """Test basic config initialization."""
        config = Config()
        assert config.project_root.exists()
        assert config.data_dir.exists()
        assert config.results_dir.exists()
        assert config.logs_dir.exists()
    
    def test_config_with_custom_env_file(self):
        """Test config with custom env file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("GROQ_API_KEY=test_key\n")
            f.write("LLM_MODEL=test_model\n")
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            assert config.groq_api_key == "test_key"
            assert config.model_config['llm']['model'] == "test_model"
        finally:
            os.unlink(temp_path)
    
    def test_api_key_validation(self):
        """Test API key validation."""
        config = Config()
        validation = config.validate_api_keys()
        assert isinstance(validation, dict)
        assert 'groq' in validation
        assert 'openai' in validation
    
    def test_model_config_structure(self):
        """Test model configuration structure."""
        config = Config()
        model_config = config.model_config
        
        # Check traditional config
        assert 'traditional' in model_config
        assert 'vectorizer' in model_config['traditional']
        assert 'classifier' in model_config['traditional']
        
        # Check LLM config
        assert 'llm' in model_config
        assert 'model' in model_config['llm']
        assert 'temperature' in model_config['llm']
        assert 'rate_limit_delay' in model_config['llm']
    
    def test_dataset_config(self):
        """Test dataset configuration."""
        config = Config()
        dataset_config = config.dataset_config
        
        assert dataset_config['name'] == 'pietrolesci/eurlex-57k'
        assert isinstance(dataset_config['train_size'], int)
        assert isinstance(dataset_config['test_size'], int)
        assert dataset_config['train_size'] > 0
    
    def test_directory_creation(self):
        """Test that directories are created properly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the project root
            config = Config()
            config.project_root = Path(tmpdir)
            config._setup_directories()
            
            assert (Path(tmpdir) / 'data').exists()
            assert (Path(tmpdir) / 'results').exists()
            assert (Path(tmpdir) / 'logs').exists() 