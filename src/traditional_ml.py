"""
Traditional machine learning classifier module.

Implements TF-IDF vectorization with Logistic Regression for text classification.
"""

import time
import logging
import joblib
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


logger = logging.getLogger(__name__)


class TraditionalClassifier:
    """
    Traditional ML classifier using TF-IDF and Logistic Regression.
    
    Optimized for legal document classification with customized vectorization
    parameters for handling domain-specific terminology.
    """
    
    def __init__(self, config):
        """
        Initialize classifier with configuration.
        
        Args:
            config: Configuration object with model settings
        """
        self.config = config
        self.model_config = config.model_config['traditional']
        
        # Initialize components
        self.vectorizer = self._create_vectorizer()
        self.classifier = self._create_classifier()
        
        # Model paths
        self.vectorizer_path = config.results_dir / "eurlex_vectorizer.pkl"
        self.model_path = config.results_dir / "eurlex_model.pkl"
        
        # Metrics storage
        self.metrics = {}
    
    def _create_vectorizer(self) -> TfidfVectorizer:
        """Create TF-IDF vectorizer with optimized settings."""
        vectorizer_config = self.model_config['vectorizer']
        return TfidfVectorizer(**vectorizer_config)
    
    def _create_classifier(self) -> LogisticRegression:
        """Create Logistic Regression classifier."""
        classifier_config = self.model_config['classifier']
        return LogisticRegression(**classifier_config)
    
    def train(self, 
              train_texts: list, 
              train_labels: list,
              validation_texts: Optional[list] = None,
              validation_labels: Optional[list] = None) -> Dict[str, float]:
        """
        Train the classifier on provided data.
        
        Args:
            train_texts: List of training texts
            train_labels: List of training labels
            validation_texts: Optional validation texts
            validation_labels: Optional validation labels
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting traditional ML training...")
        
        # Vectorization
        logger.info("Vectorizing training data...")
        start_time = time.time()
        X_train = self.vectorizer.fit_transform(train_texts)
        vectorization_time = time.time() - start_time
        
        logger.info(f"Training matrix shape: {X_train.shape}")
        logger.info(f"Matrix density: {X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.4f}")
        
        # Training
        logger.info("Training Logistic Regression model...")
        start_time = time.time()
        self.classifier.fit(X_train, train_labels)
        training_time = time.time() - start_time
        
        # Calculate training metrics
        train_predictions = self.classifier.predict(X_train)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        
        metrics = {
            'vectorization_time': vectorization_time,
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'n_features': X_train.shape[1]
        }
        
        # Validation if provided
        if validation_texts is not None and validation_labels is not None:
            val_metrics = self.evaluate(validation_texts, validation_labels)
            metrics['validation_accuracy'] = val_metrics['accuracy']
            metrics['validation_f1'] = val_metrics['f1_score']
        
        self.metrics['training'] = metrics
        logger.info(f"Training completed in {training_time:.2f}s")
        
        return metrics
    
    def predict(self, texts: list) -> np.ndarray:
        """
        Predict labels for given texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            Array of predicted labels
        """
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)
    
    def predict_proba(self, texts: list) -> np.ndarray:
        """
        Predict class probabilities for given texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            Array of class probabilities
        """
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)
    
    def evaluate(self, 
                 test_texts: list, 
                 test_labels: list,
                 label_names: Optional[list] = None) -> Dict[str, Any]:
        """
        Evaluate classifier performance on test data.
        
        Args:
            test_texts: List of test texts
            test_labels: List of true labels
            label_names: Optional list of label names
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating traditional classifier...")
        
        # Predictions
        start_time = time.time()
        predictions = self.predict(test_texts)
        probabilities = self.predict_proba(test_texts)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average='macro')
        precision = precision_score(test_labels, predictions, average='macro')
        recall = recall_score(test_labels, predictions, average='macro')
        
        # Detailed classification report
        report = classification_report(
            test_labels, 
            predictions,
            target_names=label_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        
        # Performance metrics
        avg_inference_time = inference_time / len(test_texts)
        throughput = int(1 / avg_inference_time)
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'total_inference_time': inference_time,
            'avg_inference_time': avg_inference_time,
            'throughput_per_second': throughput,
            'test_size': len(test_texts)
        }
        
        # Add class-specific metrics if binary classification
        if len(set(test_labels)) == 2 and label_names:
            for i, class_name in enumerate(label_names):
                if class_name in report:
                    metrics[f'{class_name.lower().replace(" ", "_")}_f1'] = report[class_name]['f1-score']
                    metrics[f'{class_name.lower().replace(" ", "_")}_precision'] = report[class_name]['precision']
                    metrics[f'{class_name.lower().replace(" ", "_")}_recall'] = report[class_name]['recall']
        
        self.metrics['evaluation'] = metrics
        
        logger.info(f"Evaluation completed - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return metrics
    
    def save_models(self) -> None:
        """Save trained vectorizer and classifier to disk."""
        logger.info(f"Saving vectorizer to {self.vectorizer_path}")
        joblib.dump(self.vectorizer, self.vectorizer_path)
        
        logger.info(f"Saving model to {self.model_path}")
        joblib.dump(self.classifier, self.model_path)
    
    def load_models(self) -> None:
        """Load trained vectorizer and classifier from disk."""
        if self.vectorizer_path.exists():
            logger.info(f"Loading vectorizer from {self.vectorizer_path}")
            self.vectorizer = joblib.load(self.vectorizer_path)
        
        if self.model_path.exists():
            logger.info(f"Loading model from {self.model_path}")
            self.classifier = joblib.load(self.model_path)
    
    def get_feature_importance(self, n_features: int = 20) -> Dict[str, float]:
        """
        Get top N most important features for each class.
        
        Args:
            n_features: Number of top features to return
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(self.classifier, 'coef_'):
            logger.warning("Model not trained yet")
            return {}
        
        feature_names = self.vectorizer.get_feature_names_out()
        coef = self.classifier.coef_[0]
        
        # Get top positive and negative features
        top_positive_idx = np.argsort(coef)[-n_features:][::-1]
        top_negative_idx = np.argsort(coef)[:n_features]
        
        feature_importance = {
            'positive_class': {
                feature_names[i]: float(coef[i]) 
                for i in top_positive_idx
            },
            'negative_class': {
                feature_names[i]: float(coef[i]) 
                for i in top_negative_idx
            }
        }
        
        return feature_importance 