"""
Text Classification Analysis Package

A comprehensive framework for comparing traditional ML and LLM approaches
to text classification using the EurLex-57K legal documents dataset.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "contact@example.com"

from .config import Config
from .data_loader import DataLoader
from .traditional_ml import TraditionalClassifier
from .llm_classifier import LLMClassifier
from .evaluation import Evaluator
from .visualization import Visualizer

__all__ = [
    "Config",
    "DataLoader",
    "TraditionalClassifier",
    "LLMClassifier",
    "Evaluator",
    "Visualizer",
] 