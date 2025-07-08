"""
Visualization module for classification results.

Creates charts and visual representations of model performance.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


logger = logging.getLogger(__name__)


class Visualizer:
    """
    Visualizer for classification results and comparisons.
    
    Generates various plots and charts for analysis.
    """
    
    def __init__(self, config):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.output_dir = config.results_dir / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self,
                            y_true: List[int],
                            y_pred: List[int],
                            labels: List[str],
                            title: str = "Confusion Matrix",
                            filename: Optional[str] = None) -> Path:
        """
        Plot confusion matrix for classification results.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class label names
            title: Plot title
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if not filename:
            filename = f"{title.lower().replace(' ', '_')}.png"
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {filepath}")
        return filepath
    
    def plot_performance_comparison(self,
                                  comparison_data: Dict[str, Any],
                                  filename: str = "performance_comparison.png") -> Path:
        """
        Plot performance comparison across methods.
        
        Args:
            comparison_data: Dictionary with method comparisons
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        methods = comparison_data.get('methods', {})
        
        if not methods:
            logger.warning("No methods to compare")
            return None
        
        # Extract metrics
        method_names = []
        accuracies = []
        f1_scores = []
        
        for method, metrics in methods.items():
            method_names.append(method.replace('_', ' ').title())
            accuracies.append(metrics.get('accuracy', 0))
            f1_scores.append(metrics.get('f1_score', 0))
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy plot
        bars1 = ax1.bar(method_names, accuracies, alpha=0.8)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Classification Accuracy by Method')
        ax1.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.2%}', ha='center', va='bottom')
        
        # F1-Score plot
        bars2 = ax2.bar(method_names, f1_scores, alpha=0.8)
        ax2.set_ylabel('F1-Score')
        ax2.set_title('F1-Score by Method')
        ax2.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, f1 in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved performance comparison to {filepath}")
        return filepath
    
    def plot_speed_cost_analysis(self,
                               comparison_data: Dict[str, Any],
                               filename: str = "speed_cost_analysis.png") -> Path:
        """
        Plot speed vs cost analysis.
        
        Args:
            comparison_data: Dictionary with method comparisons
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        methods = comparison_data.get('methods', {})
        
        if not methods:
            logger.warning("No methods to compare")
            return None
        
        # Extract metrics
        method_names = []
        speeds = []
        costs = []
        accuracies = []
        
        for method, metrics in methods.items():
            method_names.append(method.replace('_', ' ').title())
            
            # Speed (docs per second)
            if 'throughput_per_sec' in metrics:
                speeds.append(metrics['throughput_per_sec'])
            else:
                # Convert from response time
                avg_time = metrics.get('avg_response_time_s', 1)
                speeds.append(1 / avg_time if avg_time > 0 else 0)
            
            costs.append(metrics.get('cost_per_1k', 0))
            accuracies.append(metrics.get('accuracy', 0))
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use accuracy for bubble size
        sizes = [acc * 1000 for acc in accuracies]
        
        scatter = ax.scatter(speeds, costs, s=sizes, alpha=0.6, edgecolors='black')
        
        # Add labels
        for i, name in enumerate(method_names):
            ax.annotate(name, (speeds[i], costs[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Processing Speed (docs/second)', fontsize=12)
        ax.set_ylabel('Cost per 1000 documents ($)', fontsize=12)
        ax.set_title('Speed vs Cost Analysis (bubble size = accuracy)', fontsize=14)
        
        # Use log scale for x-axis due to large differences
        ax.set_xscale('log')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend for bubble sizes
        legend_sizes = [0.7, 0.8, 0.9]
        legend_bubbles = []
        for size in legend_sizes:
            legend_bubbles.append(plt.scatter([], [], s=size*1000, alpha=0.6, 
                                            edgecolors='black', label=f'{size:.0%} accuracy'))
        ax.legend(handles=legend_bubbles, loc='upper right')
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved speed-cost analysis to {filepath}")
        return filepath
    
    def plot_feature_importance(self,
                              feature_importance: Dict[str, Dict[str, float]],
                              n_features: int = 15,
                              filename: str = "feature_importance.png") -> Path:
        """
        Plot top feature importance for traditional ML.
        
        Args:
            feature_importance: Dictionary with positive/negative class features
            n_features: Number of top features to show
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        if not feature_importance:
            logger.warning("No feature importance data")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Positive class features
        if 'positive_class' in feature_importance:
            pos_features = list(feature_importance['positive_class'].items())[:n_features]
            pos_words, pos_scores = zip(*pos_features)
            
            ax1.barh(range(len(pos_words)), pos_scores, alpha=0.8)
            ax1.set_yticks(range(len(pos_words)))
            ax1.set_yticklabels(pos_words)
            ax1.set_xlabel('Feature Importance')
            ax1.set_title('Top Features for Complex Regulatory Class')
            ax1.invert_yaxis()
        
        # Negative class features
        if 'negative_class' in feature_importance:
            neg_features = list(feature_importance['negative_class'].items())[:n_features]
            neg_words, neg_scores = zip(*neg_features)
            
            ax2.barh(range(len(neg_words)), neg_scores, alpha=0.8)
            ax2.set_yticks(range(len(neg_words)))
            ax2.set_yticklabels(neg_words)
            ax2.set_xlabel('Feature Importance')
            ax2.set_title('Top Features for Simple Administrative Class')
            ax2.invert_yaxis()
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature importance to {filepath}")
        return filepath
    
    def plot_class_distribution(self,
                              train_labels: List[int],
                              test_labels: List[int],
                              label_names: List[str],
                              filename: str = "class_distribution.png") -> Path:
        """
        Plot class distribution in train and test sets.
        
        Args:
            train_labels: Training labels
            test_labels: Test labels
            label_names: Class names
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Training set distribution
        train_counts = np.bincount(train_labels)
        ax1.pie(train_counts, labels=label_names, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Training Set Distribution')
        
        # Test set distribution
        test_counts = np.bincount(test_labels)
        ax2.pie(test_counts, labels=label_names, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Test Set Distribution')
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved class distribution to {filepath}")
        return filepath
    
    def create_summary_report(self, 
                            comparison_data: Dict[str, Any],
                            output_format: str = "png") -> List[Path]:
        """
        Create a comprehensive visual summary report.
        
        Args:
            comparison_data: Comparison results
            output_format: Output format (png, pdf)
            
        Returns:
            List of paths to generated visualizations
        """
        generated_files = []
        
        # Performance comparison
        perf_plot = self.plot_performance_comparison(comparison_data)
        if perf_plot:
            generated_files.append(perf_plot)
        
        # Speed-cost analysis
        speed_plot = self.plot_speed_cost_analysis(comparison_data)
        if speed_plot:
            generated_files.append(speed_plot)
        
        logger.info(f"Generated {len(generated_files)} visualization files")
        return generated_files 