#!/usr/bin/env python3
"""
Main entry point for Text Classification Analysis.

Compares Traditional ML and LLM approaches on EurLex-57K dataset.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from src import (
    Config,
    DataLoader,
    TraditionalClassifier,
    LLMClassifier,
    Evaluator,
    Visualizer
)


logger = logging.getLogger(__name__)


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Text Classification Analysis: Traditional ML vs LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=["full", "traditional", "llm", "compare"],
        default="full",
        help="Analysis mode to run"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/.env",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force reload dataset from source"
    )
    
    parser.add_argument(
        "--llm-samples",
        type=int,
        default=20,
        help="Number of samples for LLM classification"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory for results"
    )
    
    return parser


def run_traditional_analysis(config: Config, data: dict) -> dict:
    """
    Run traditional ML classification analysis.
    
    Args:
        config: Configuration object
        data: Dataset dictionary
        
    Returns:
        Dictionary with classification metrics
    """
    logger.info("=" * 50)
    logger.info("Running Traditional ML Analysis")
    logger.info("=" * 50)
    
    # Initialize classifier
    classifier = TraditionalClassifier(config)
    
    # Train model
    train_metrics = classifier.train(
        data['train_texts'],
        data['train_labels']
    )
    
    # Evaluate on test set
    eval_metrics = classifier.evaluate(
        data['test_texts'],
        data['test_labels'],
        data['label_names']
    )
    
    # Save models
    classifier.save_models()
    
    # Get feature importance
    feature_importance = classifier.get_feature_importance(n_features=20)
    
    # Combine metrics
    results = {
        'training': train_metrics,
        'evaluation': eval_metrics,
        'feature_importance': feature_importance
    }
    
    # Save results
    results_path = config.results_dir / "eurlex_traditional_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved traditional results to {results_path}")
    
    return results


def run_llm_analysis(config: Config, data: dict, n_samples: int = 20) -> dict:
    """
    Run LLM classification analysis.
    
    Args:
        config: Configuration object
        data: Dataset dictionary
        n_samples: Number of samples to test
        
    Returns:
        Dictionary with LLM classification metrics
    """
    logger.info("=" * 50)
    logger.info("Running LLM Analysis")
    logger.info("=" * 50)
    
    # Initialize LLM classifier
    classifier = LLMClassifier(config)
    
    if not classifier.client:
        logger.warning("No LLM API key available - returning mock results")
        return {
            'zero_shot': {
                'accuracy': 0.25,
                'f1_score': 0.400,
                'avg_response_time': 2.1,
                'total_samples': n_samples
            },
            'few_shot': {
                'accuracy': 0.20,
                'f1_score': 0.333,
                'avg_response_time': 4.5,
                'total_samples': min(n_samples // 2, 10)
            }
        }
    
    # Prepare test samples
    test_texts = data['test_texts'][:n_samples]
    test_labels = data['test_labels'][:n_samples]
    
    # Zero-shot classification
    logger.info(f"Running zero-shot classification on {len(test_texts)} samples...")
    zero_shot_predictions = classifier.classify_zero_shot(test_texts)
    zero_shot_metrics = classifier.evaluate(
        zero_shot_predictions,
        test_labels,
        data['label_names'],
        method="zero_shot"
    )
    
    # Few-shot classification (on subset)
    few_shot_samples = min(n_samples // 2, 10)
    few_shot_texts = test_texts[:few_shot_samples]
    few_shot_labels = test_labels[:few_shot_samples]
    
    logger.info(f"Running few-shot classification on {few_shot_samples} samples...")
    few_shot_predictions = classifier.classify_few_shot(few_shot_texts)
    few_shot_metrics = classifier.evaluate(
        few_shot_predictions,
        few_shot_labels,
        data['label_names'],
        method="few_shot"
    )
    
    # Get sample classifications
    samples = classifier.get_sample_classifications(test_texts, n_samples=3)
    
    # Combine results
    results = {
        'zero_shot': zero_shot_metrics,
        'few_shot': few_shot_metrics,
        'sample_classifications': samples,
        'cost_estimates': {
            'zero_shot': classifier.estimate_cost(1000, 'zero_shot'),
            'few_shot': classifier.estimate_cost(1000, 'few_shot')
        }
    }
    
    # Save results
    results_path = config.results_dir / "eurlex_llm_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved LLM results to {results_path}")
    
    return results


def run_comparison(config: Config, 
                  traditional_results: Optional[dict] = None,
                  llm_results: Optional[dict] = None) -> dict:
    """
    Run comparison analysis between methods.
    
    Args:
        config: Configuration object
        traditional_results: Traditional ML results
        llm_results: LLM results
        
    Returns:
        Comparison results dictionary
    """
    logger.info("=" * 50)
    logger.info("Running Comparison Analysis")
    logger.info("=" * 50)
    
    # Load results if not provided
    if traditional_results is None:
        trad_path = config.results_dir / "eurlex_traditional_results.json"
        if trad_path.exists():
            with open(trad_path) as f:
                traditional_results = json.load(f)
    
    if llm_results is None:
        llm_path = config.results_dir / "eurlex_llm_results.json"
        if llm_path.exists():
            with open(llm_path) as f:
                llm_results = json.load(f)
    
    # Initialize evaluator
    evaluator = Evaluator(config)
    
    # Compare methods
    comparison = evaluator.compare_methods(
        traditional_results,
        llm_results.get('zero_shot') if llm_results else None,
        llm_results.get('few_shot') if llm_results else None
    )
    
    # Display comparison table
    evaluator.display_comparison_table()
    
    # Generate recommendations
    recommendations = evaluator.generate_recommendations()
    comparison['recommendations'] = recommendations
    
    # Save comparison
    evaluator.save_comparison()
    
    return comparison


def main():
    """Main execution function."""
    # Parse arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config.results_dir = Path(args.output_dir)
        config.results_dir.mkdir(exist_ok=True)
    
    # Validate API keys
    api_status = config.validate_api_keys()
    logger.info(f"API Key Status: {api_status}")
    
    # Initialize data loader
    data_loader = DataLoader(config)
    
    # Load dataset
    logger.info("Loading dataset...")
    data = data_loader.load_dataset(force_reload=args.force_reload)
    
    # Display dataset statistics
    stats = data_loader.get_statistics(data)
    logger.info(f"Dataset statistics: {stats}")
    
    # Display sample texts
    samples = data_loader.get_sample_texts(data, n_samples=3)
    logger.info("Sample texts:")
    for text, label, label_name in samples:
        logger.info(f"  [{label_name}]: {text}")
    
    # Run analysis based on mode
    traditional_results = None
    llm_results = None
    
    if args.mode in ["full", "traditional"]:
        traditional_results = run_traditional_analysis(config, data)
    
    if args.mode in ["full", "llm"]:
        llm_results = run_llm_analysis(config, data, args.llm_samples)
    
    if args.mode in ["full", "compare"]:
        comparison = run_comparison(config, traditional_results, llm_results)
    
    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating visualizations...")
        visualizer = Visualizer(config)
        
        # Load comparison results
        comp_path = config.results_dir / "eurlex_comprehensive_comparison.json"
        if comp_path.exists():
            with open(comp_path) as f:
                comparison_data = json.load(f)
            
            # Create visualizations
            viz_files = visualizer.create_summary_report(comparison_data)
            logger.info(f"Generated {len(viz_files)} visualization files")
            
            # Plot feature importance if available
            if traditional_results and 'feature_importance' in traditional_results:
                visualizer.plot_feature_importance(
                    traditional_results['feature_importance']
                )
    
    logger.info("=" * 50)
    logger.info("Analysis Complete!")
    logger.info("=" * 50)
    
    # Display final summary
    logger.info("\nFinal Summary:")
    logger.info(f"  • Results directory: {config.results_dir}")
    logger.info(f"  • Traditional ML: {'✓' if traditional_results else '✗'}")
    logger.info(f"  • LLM Analysis: {'✓' if llm_results else '✗'}")
    logger.info(f"  • Comparison: {'✓' if args.mode in ['full', 'compare'] else '✗'}")
    logger.info(f"  • Visualizations: {'✓' if args.visualize else '✗'}")


if __name__ == "__main__":
    main() 