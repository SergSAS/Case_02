"""
Data loader module for EurLex-57K dataset.

Handles dataset loading, preprocessing, and splitting for text classification.
"""

import pickle
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from datasets import load_dataset


logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader for EurLex-57K legal documents dataset.
    
    Provides methods for loading, preprocessing, and splitting the dataset.
    """
    
    def __init__(self, config):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Configuration object with dataset settings
        """
        self.config = config
        self.dataset_config = config.dataset_config
        self.cache_file = config.data_dir / "eurlex_sample.pkl"
    
    def load_dataset(self, force_reload: bool = False) -> Dict[str, any]:
        """
        Load EurLex-57K dataset from cache or download.
        
        Args:
            force_reload: Force download even if cache exists
            
        Returns:
            Dictionary containing train/test texts and labels
        """
        if self.cache_file.exists() and not force_reload:
            logger.info("Loading dataset from cache...")
            return self._load_from_cache()
        
        logger.info("Downloading EurLex-57K dataset...")
        return self._download_and_process()
    
    def _load_from_cache(self) -> Dict[str, any]:
        """Load preprocessed dataset from cache file."""
        with open(self.cache_file, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded {len(data['train_texts'])} training samples from cache")
        return data
    
    def _download_and_process(self) -> Dict[str, any]:
        """Download and process EurLex dataset."""
        # Load dataset with specified sizes
        train_size = self.dataset_config["train_size"]
        val_size = self.dataset_config["validation_size"]
        test_size = self.dataset_config["test_size"]
        
        split_str = f"train[:{train_size}]+validation[:{val_size}]+test[:{test_size}]"
        dataset = load_dataset(self.dataset_config["name"], split=split_str)
        
        # Process into binary classification
        processed_data = self._create_binary_classification(dataset)
        
        # Cache the processed data
        self._save_to_cache(processed_data)
        
        return processed_data
    
    def _create_binary_classification(self, dataset) -> Dict[str, any]:
        """
        Create binary classification task from EurLex dataset.
        
        Classifies documents as:
        - Complex Regulatory (>2000 chars): label 1
        - Simple Administrative (<=2000 chars): label 0
        
        Args:
            dataset: HuggingFace dataset object
            
        Returns:
            Dictionary with train/test splits
        """
        train_texts = []
        train_labels = []
        test_texts = []
        test_labels = []
        
        threshold = self.dataset_config["classification_threshold"]
        train_size = 4000  # 80% for training
        
        logger.info(f"Processing {len(dataset)} documents...")
        
        for i, item in enumerate(dataset):
            text = item['text']
            # Binary classification based on document length
            label = 1 if len(text) > threshold else 0
            
            if i < train_size:
                train_texts.append(text)
                train_labels.append(label)
            else:
                test_texts.append(text)
                test_labels.append(label)
        
        # Calculate class distribution
        train_dist = self._calculate_distribution(train_labels)
        test_dist = self._calculate_distribution(test_labels)
        
        logger.info(f"Training set distribution: {train_dist}")
        logger.info(f"Test set distribution: {test_dist}")
        
        return {
            'train_texts': train_texts,
            'train_labels': train_labels,
            'test_texts': test_texts,
            'test_labels': test_labels,
            'label_names': ['Simple Administrative', 'Complex Regulatory'],
            'dataset_info': {
                'name': 'EurLex-57K',
                'task': 'Legal document classification',
                'domain': 'European Union legal documents',
                'train_distribution': train_dist,
                'test_distribution': test_dist
            }
        }
    
    def _calculate_distribution(self, labels: List[int]) -> Dict[str, float]:
        """Calculate class distribution percentages."""
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        return {
            'class_0': counts[0] / total * 100,
            'class_1': counts[1] / total * 100
        }
    
    def _save_to_cache(self, data: Dict[str, any]) -> None:
        """Save processed data to cache file."""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Cached dataset to {self.cache_file}")
    
    def get_sample_texts(self, 
                        data: Dict[str, any], 
                        n_samples: int = 3,
                        from_set: str = 'train') -> List[Tuple[str, int, str]]:
        """
        Get sample texts for display or analysis.
        
        Args:
            data: Dataset dictionary
            n_samples: Number of samples to return
            from_set: 'train' or 'test'
            
        Returns:
            List of tuples (text_preview, label, label_name)
        """
        texts = data[f'{from_set}_texts']
        labels = data[f'{from_set}_labels']
        label_names = data['label_names']
        
        samples = []
        for i in range(min(n_samples, len(texts))):
            text_preview = texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i]
            label = labels[i]
            label_name = label_names[label]
            samples.append((text_preview, label, label_name))
        
        return samples
    
    def get_statistics(self, data: Dict[str, any]) -> Dict[str, any]:
        """
        Calculate dataset statistics.
        
        Args:
            data: Dataset dictionary
            
        Returns:
            Dictionary with various statistics
        """
        train_lengths = [len(text) for text in data['train_texts']]
        test_lengths = [len(text) for text in data['test_texts']]
        
        # Calculate distributions if not present
        train_dist = data.get('dataset_info', {}).get('train_distribution')
        test_dist = data.get('dataset_info', {}).get('test_distribution')
        
        if not train_dist:
            train_dist = self._calculate_distribution(data['train_labels'])
        if not test_dist:
            test_dist = self._calculate_distribution(data['test_labels'])
        
        return {
            'train_size': len(data['train_texts']),
            'test_size': len(data['test_texts']),
            'avg_train_length': np.mean(train_lengths),
            'avg_test_length': np.mean(test_lengths),
            'max_train_length': np.max(train_lengths),
            'max_test_length': np.max(test_lengths),
            'train_distribution': train_dist,
            'test_distribution': test_dist
        } 