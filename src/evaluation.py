"""
Evaluation module for comparing classification methods.

Provides comprehensive comparison and analysis of traditional ML and LLM approaches.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table


logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for comparing different classification approaches.
    
    Generates comprehensive comparison reports and recommendations.
    """
    
    def __init__(self, config):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.console = Console()
        self.comparison_results = {}
    
    def compare_methods(self,
                       traditional_metrics: Dict[str, Any],
                       llm_zero_shot_metrics: Optional[Dict[str, Any]] = None,
                       llm_few_shot_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare performance across different classification methods.
        
        Args:
            traditional_metrics: Metrics from traditional ML
            llm_zero_shot_metrics: Metrics from LLM zero-shot
            llm_few_shot_metrics: Metrics from LLM few-shot
            
        Returns:
            Comprehensive comparison dictionary
        """
        logger.info("Generating comprehensive comparison...")
        
        comparison = {
            'methods': {},
            'best_accuracy': {'method': None, 'value': 0},
            'best_f1': {'method': None, 'value': 0},
            'fastest': {'method': None, 'value': float('inf')},
            'most_cost_effective': {'method': None, 'value': 0}
        }
        
        # Traditional ML metrics
        if traditional_metrics:
            trad_summary = self._summarize_traditional(traditional_metrics)
            comparison['methods']['traditional_ml'] = trad_summary
            self._update_best_metrics(comparison, 'traditional_ml', trad_summary)
        
        # LLM Zero-shot metrics
        if llm_zero_shot_metrics:
            zero_shot_summary = self._summarize_llm(llm_zero_shot_metrics, 'zero_shot')
            comparison['methods']['llm_zero_shot'] = zero_shot_summary
            self._update_best_metrics(comparison, 'llm_zero_shot', zero_shot_summary)
        
        # LLM Few-shot metrics
        if llm_few_shot_metrics:
            few_shot_summary = self._summarize_llm(llm_few_shot_metrics, 'few_shot')
            comparison['methods']['llm_few_shot'] = few_shot_summary
            self._update_best_metrics(comparison, 'llm_few_shot', few_shot_summary)
        
        # Generate insights
        comparison['insights'] = self._generate_insights(comparison)
        
        # Store results
        self.comparison_results = comparison
        
        return comparison
    
    def _summarize_traditional(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize traditional ML metrics."""
        eval_metrics = metrics.get('evaluation', {})
        
        return {
            'accuracy': eval_metrics.get('accuracy', 0),
            'f1_score': eval_metrics.get('f1_score', 0),
            'precision': eval_metrics.get('precision', 0),
            'recall': eval_metrics.get('recall', 0),
            'inference_time_ms': eval_metrics.get('avg_inference_time', 0) * 1000,
            'throughput_per_sec': eval_metrics.get('throughput_per_second', 0),
            'cost_per_1k': 0,  # Traditional ML has no API cost
            'training_time': metrics.get('training', {}).get('training_time', 0),
            'model_size_features': metrics.get('training', {}).get('n_features', 0)
        }
    
    def _summarize_llm(self, metrics: Dict[str, Any], method: str) -> Dict[str, Any]:
        """Summarize LLM metrics."""
        return {
            'accuracy': metrics.get('accuracy', 0),
            'f1_score': metrics.get('f1_score', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'avg_response_time_s': metrics.get('avg_response_time', 0),
            'throughput_per_sec': 1 / metrics.get('avg_response_time', 1) if metrics.get('avg_response_time', 0) > 0 else 0,
            'cost_per_1k': 0.30 if method == 'zero_shot' else 0.50,  # Estimated costs
            'total_samples': metrics.get('total_samples', 0),
            'method': method
        }
    
    def _update_best_metrics(self, comparison: Dict, method: str, summary: Dict) -> None:
        """Update best performing metrics."""
        # Best accuracy
        if summary.get('accuracy', 0) > comparison['best_accuracy']['value']:
            comparison['best_accuracy'] = {'method': method, 'value': summary['accuracy']}
        
        # Best F1
        if summary.get('f1_score', 0) > comparison['best_f1']['value']:
            comparison['best_f1'] = {'method': method, 'value': summary['f1_score']}
        
        # Fastest (lowest response time)
        response_time = summary.get('inference_time_ms', summary.get('avg_response_time_s', 0) * 1000)
        if response_time < comparison['fastest']['value'] and response_time > 0:
            comparison['fastest'] = {'method': method, 'value': response_time}
        
        # Most cost effective (accuracy per dollar)
        cost = summary.get('cost_per_1k', 0)
        if cost == 0:  # Traditional ML
            comparison['most_cost_effective'] = {'method': method, 'value': 'free'}
        elif cost > 0 and summary.get('accuracy', 0) > 0:
            cost_effectiveness = summary['accuracy'] / cost
            if isinstance(comparison['most_cost_effective']['value'], str) or \
               cost_effectiveness > comparison['most_cost_effective'].get('value', 0):
                comparison['most_cost_effective'] = {'method': method, 'value': cost_effectiveness}
    
    def _generate_insights(self, comparison: Dict) -> List[str]:
        """Generate insights from comparison."""
        insights = []
        
        # Performance insight
        best_acc = comparison['best_accuracy']
        if best_acc['method'] == 'traditional_ml':
            insights.append(
                f"Traditional ML achieves the best accuracy ({best_acc['value']:.2%}), "
                "demonstrating that classical approaches remain highly effective for structured text classification."
            )
        else:
            insights.append(
                f"LLM ({best_acc['method']}) achieves superior accuracy ({best_acc['value']:.2%}), "
                "leveraging contextual understanding for better classification."
            )
        
        # Speed insight
        fastest = comparison['fastest']
        if fastest['method'] == 'traditional_ml':
            trad_throughput = comparison['methods']['traditional_ml']['throughput_per_sec']
            insights.append(
                f"Traditional ML is dramatically faster ({trad_throughput:.0f} docs/sec), "
                "making it ideal for high-volume production environments."
            )
        
        # Cost insight
        if comparison['most_cost_effective']['value'] == 'free':
            insights.append(
                "Traditional ML offers zero operational costs after training, "
                "providing unlimited scalability without API expenses."
            )
        
        # Trade-off insight
        insights.append(
            "The choice between approaches depends on specific requirements: "
            "Traditional ML for speed and cost-efficiency, "
            "LLM for flexibility and minimal training data needs."
        )
        
        return insights
    
    def display_comparison_table(self) -> None:
        """Display comparison results in a formatted table."""
        if not self.comparison_results:
            logger.warning("No comparison results available")
            return
        
        # Create comparison table
        table = Table(title="🏆 Classification Methods Comparison", show_header=True)
        table.add_column("Method", style="cyan", no_wrap=True)
        table.add_column("Accuracy", justify="center", style="green")
        table.add_column("F1-Score", justify="center")
        table.add_column("Speed", justify="center", style="yellow")
        table.add_column("Cost", justify="center", style="red")
        table.add_column("Scalability", justify="center")
        
        # Add rows for each method
        methods = self.comparison_results.get('methods', {})
        
        if 'traditional_ml' in methods:
            trad = methods['traditional_ml']
            table.add_row(
                "Traditional ML",
                f"{trad['accuracy']:.2%}",
                f"{trad['f1_score']:.3f}",
                f"⚡ {trad['throughput_per_sec']:.0f} docs/s",
                "💰 $0",
                "✅ Excellent"
            )
        
        if 'llm_zero_shot' in methods:
            zero = methods['llm_zero_shot']
            table.add_row(
                "LLM Zero-shot",
                f"{zero['accuracy']:.2%}",
                f"{zero['f1_score']:.3f}",
                f"🐌 {zero['avg_response_time_s']:.1f}s/doc",
                f"💸 ${zero['cost_per_1k']}/1K",
                "⚠️ Limited"
            )
        
        if 'llm_few_shot' in methods:
            few = methods['llm_few_shot']
            table.add_row(
                "LLM Few-shot",
                f"{few['accuracy']:.2%}",
                f"{few['f1_score']:.3f}",
                f"🐌 {few['avg_response_time_s']:.1f}s/doc",
                f"💸 ${few['cost_per_1k']}/1K",
                "⚠️ Limited"
            )
        
        self.console.print(table)
        
        # Print insights
        self.console.print("\n📊 Key Insights:", style="bold")
        for insight in self.comparison_results.get('insights', []):
            self.console.print(f"   • {insight}")
    
    def save_comparison(self, filename: Optional[str] = None) -> Path:
        """
        Save comparison results to JSON file.
        
        Args:
            filename: Optional filename, defaults to eurlex_comprehensive_comparison.json
            
        Returns:
            Path to saved file
        """
        if not self.comparison_results:
            logger.warning("No comparison results to save")
            return None
        
        filename = filename or "eurlex_comprehensive_comparison.json"
        filepath = self.config.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.comparison_results, f, indent=2)
        
        logger.info(f"Saved comparison results to {filepath}")
        return filepath
    
    def generate_recommendations(self) -> Dict[str, List[str]]:
        """
        Generate usage recommendations based on comparison results.
        
        Returns:
            Dictionary of recommendations by use case
        """
        if not self.comparison_results:
            return {}
        
        recommendations = {
            'prototyping': [
                "Use LLM zero-shot for quick experiments",
                "No training data required",
                "Immediate results with reasonable accuracy"
            ],
            'production': [
                "Deploy Traditional ML for high-volume processing",
                "Excellent speed and zero operational costs",
                "Stable performance with proper training data"
            ],
            'complex_classification': [
                "Consider LLM few-shot for nuanced categories",
                "Better contextual understanding",
                "Worth the extra cost for critical decisions"
            ],
            'hybrid_approach': [
                "Use Traditional ML as primary classifier",
                "Route uncertain cases to LLM for verification",
                "Balances speed, cost, and accuracy"
            ]
        }
        
        return recommendations
    
    def export_metrics_dataframe(self) -> pd.DataFrame:
        """
        Export comparison metrics as pandas DataFrame.
        
        Returns:
            DataFrame with all metrics
        """
        if not self.comparison_results:
            return pd.DataFrame()
        
        methods = self.comparison_results.get('methods', {})
        
        # Convert to DataFrame format
        data = []
        for method_name, metrics in methods.items():
            row = {'method': method_name}
            row.update(metrics)
            data.append(row)
        
        df = pd.DataFrame(data)
        return df 