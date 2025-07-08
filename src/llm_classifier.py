"""
LLM classifier module for text classification.

Implements zero-shot and few-shot classification using Large Language Models
via Groq API.
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from groq import Groq
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


logger = logging.getLogger(__name__)


class LLMClassifier:
    """
    LLM-based text classifier using Groq API.
    
    Supports both zero-shot and few-shot classification approaches
    for legal document categorization.
    """
    
    def __init__(self, config):
        """
        Initialize LLM classifier with configuration.
        
        Args:
            config: Configuration object with API keys and model settings
        """
        self.config = config
        self.llm_config = config.model_config['llm']
        
        # Initialize Groq client
        if config.groq_api_key:
            self.client = Groq(api_key=config.groq_api_key)
            self.model = self.llm_config['model']
            logger.info(f"Initialized Groq client with model: {self.model}")
        else:
            self.client = None
            logger.warning("No Groq API key found - LLM features will be limited")
        
        # Prompts for legal document classification
        self.system_prompt = self._create_system_prompt()
        self.few_shot_examples = self._create_few_shot_examples()
        
        # Metrics storage
        self.metrics = {}
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for legal document classification."""
        return """You are an expert legal document classifier specializing in European Union legislation. 
Your task is to classify legal documents as either "Complex Regulatory" or "Simple Administrative".

Complex Regulatory documents include: regulations, directives, decisions, recommendations that establish rules, procedures, or binding obligations.
Simple Administrative documents include: reports, communications, opinions, studies, informational documents.

Respond with only "Complex Regulatory" or "Simple Administrative"."""
    
    def _create_few_shot_examples(self) -> str:
        """Create few-shot examples for classification."""
        return """Examples:
Document: "Commission Regulation establishing detailed rules for the implementation of Council Directive..."
Classification: Complex Regulatory

Document: "Report on the implementation of the European Social Fund in Member States..."
Classification: Simple Administrative

Document: "Council Directive on the approximation of laws relating to environmental protection..."
Classification: Complex Regulatory"""
    
    def classify_zero_shot(self, 
                          texts: List[str], 
                          max_length: Optional[int] = None) -> List[int]:
        """
        Perform zero-shot classification on texts.
        
        Args:
            texts: List of texts to classify
            max_length: Maximum text length to send to API
            
        Returns:
            List of predicted labels (0 or 1)
        """
        if not self.client:
            logger.error("No API client available")
            return [0] * len(texts)
        
        predictions = []
        response_times = []
        max_length = max_length or self.config.dataset_config['text_max_length']
        
        logger.info(f"Starting zero-shot classification for {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            try:
                # Truncate text if needed
                text_sample = text[:max_length] + "..." if len(text) > max_length else text
                
                # API call
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Classify this document:\n\n{text_sample}"}
                    ],
                    max_tokens=self.llm_config['max_tokens'],
                    temperature=self.llm_config['temperature']
                )
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                # Parse response
                prediction_text = response.choices[0].message.content.strip()
                label = 1 if "Complex Regulatory" in prediction_text else 0
                predictions.append(label)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)} texts")
                
                # Rate limiting
                time.sleep(self.llm_config['rate_limit_delay'])
                
            except Exception as e:
                logger.error(f"Error classifying text {i}: {e}")
                predictions.append(0)
                response_times.append(0)
                time.sleep(self.llm_config['rate_limit_delay'] * 2)
        
        # Store metrics
        self.metrics['zero_shot_times'] = response_times
        self.metrics['zero_shot_avg_time'] = np.mean(response_times) if response_times else 0
        
        return predictions
    
    def classify_few_shot(self, 
                         texts: List[str], 
                         max_length: Optional[int] = None) -> List[int]:
        """
        Perform few-shot classification on texts.
        
        Args:
            texts: List of texts to classify
            max_length: Maximum text length to send to API
            
        Returns:
            List of predicted labels (0 or 1)
        """
        if not self.client:
            logger.error("No API client available")
            return [0] * len(texts)
        
        predictions = []
        response_times = []
        max_length = max_length or (self.config.dataset_config['text_max_length'] - 200)
        
        logger.info(f"Starting few-shot classification for {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            try:
                # Truncate text if needed
                text_sample = text[:max_length] + "..." if len(text) > max_length else text
                
                # API call with examples
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"{self.few_shot_examples}\n\nNow classify:\n{text_sample}"}
                    ],
                    max_tokens=self.llm_config['max_tokens'],
                    temperature=self.llm_config['temperature']
                )
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                # Parse response
                prediction_text = response.choices[0].message.content.strip()
                label = 1 if "Complex Regulatory" in prediction_text else 0
                predictions.append(label)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)} texts")
                
                # Rate limiting
                time.sleep(self.llm_config['rate_limit_delay'] * 1.5)
                
            except Exception as e:
                logger.error(f"Error classifying text {i}: {e}")
                predictions.append(0)
                response_times.append(0)
                time.sleep(self.llm_config['rate_limit_delay'] * 3)
        
        # Store metrics
        self.metrics['few_shot_times'] = response_times
        self.metrics['few_shot_avg_time'] = np.mean(response_times) if response_times else 0
        
        return predictions
    
    def evaluate(self, 
                predictions: List[int], 
                true_labels: List[int],
                label_names: Optional[List[str]] = None,
                method: str = "zero_shot") -> Dict[str, Any]:
        """
        Evaluate classification performance.
        
        Args:
            predictions: List of predicted labels
            true_labels: List of true labels
            label_names: Optional list of label names
            method: Classification method used
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {method} classification...")
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='macro')
        precision = precision_score(true_labels, predictions, average='macro')
        recall = recall_score(true_labels, predictions, average='macro')
        
        # Detailed report
        report = classification_report(
            true_labels,
            predictions,
            target_names=label_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Response time metrics
        avg_time_key = f"{method}_avg_time"
        avg_response_time = self.metrics.get(avg_time_key, 0)
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'avg_response_time': avg_response_time,
            'total_samples': len(predictions),
            'method': method
        }
        
        # Add class-specific metrics
        if label_names and len(label_names) == 2:
            for class_name in label_names:
                if class_name in report:
                    safe_name = class_name.lower().replace(" ", "_")
                    metrics[f'{safe_name}_f1'] = report[class_name]['f1-score']
                    metrics[f'{safe_name}_precision'] = report[class_name]['precision']
                    metrics[f'{safe_name}_recall'] = report[class_name]['recall']
        
        self.metrics[f'{method}_evaluation'] = metrics
        
        logger.info(f"{method} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return metrics
    
    def estimate_cost(self, n_texts: int, method: str = "zero_shot") -> Dict[str, float]:
        """
        Estimate API cost for classification.
        
        Args:
            n_texts: Number of texts to classify
            method: Classification method
            
        Returns:
            Dictionary with cost estimates
        """
        # Average tokens per request (rough estimate)
        avg_tokens_per_request = {
            "zero_shot": 500,  # System prompt + text sample
            "few_shot": 800    # System prompt + examples + text sample
        }
        
        tokens = avg_tokens_per_request.get(method, 500) * n_texts
        
        # Groq pricing (as of 2025)
        cost_per_1k_tokens = 0.0003  # $0.30 per 1M tokens
        estimated_cost = (tokens / 1000) * cost_per_1k_tokens
        
        return {
            'total_tokens': tokens,
            'estimated_cost_usd': estimated_cost,
            'cost_per_1k_texts': (estimated_cost / n_texts) * 1000
        }
    
    def get_sample_classifications(self, 
                                  texts: List[str], 
                                  n_samples: int = 3) -> List[Dict[str, str]]:
        """
        Get sample classifications with explanations.
        
        Args:
            texts: List of texts to classify
            n_samples: Number of samples to process
            
        Returns:
            List of dictionaries with text, prediction, and explanation
        """
        if not self.client:
            return []
        
        samples = []
        
        for i in range(min(n_samples, len(texts))):
            text = texts[i]
            text_preview = text[:200] + "..." if len(text) > 200 else text
            
            try:
                # Get classification with explanation
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Classify this document and briefly explain why:\n\n{text[:1000]}"}
                    ],
                    max_tokens=50,
                    temperature=self.llm_config['temperature']
                )
                
                classification = response.choices[0].message.content.strip()
                
                samples.append({
                    'text_preview': text_preview,
                    'classification': classification,
                    'length': len(text)
                })
                
                time.sleep(self.llm_config['rate_limit_delay'])
                
            except Exception as e:
                logger.error(f"Error getting sample classification: {e}")
                samples.append({
                    'text_preview': text_preview,
                    'classification': "Error",
                    'length': len(text)
                })
        
        return samples 