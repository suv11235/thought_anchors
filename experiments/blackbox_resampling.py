"""
Black-box resampling experiment for Thought Anchors.
"""

import os
import json
import argparse
from typing import List, Dict, Any
from datetime import datetime

from src.blackbox import BlackboxResampler, ImportanceAnalyzer
from src.sentence_analysis import SentenceAnalyzer
from config.models import ModelConfig
from config.experiments import ExperimentConfig

def run_blackbox_experiment(
    reasoning_trace: str,
    model_config: ModelConfig,
    experiment_config: ExperimentConfig,
    output_dir: str = "./results"
) -> Dict[str, Any]:
    """
    Run black-box resampling experiment.
    
    Args:
        reasoning_trace: The reasoning trace to analyze
        model_config: Model configuration
        experiment_config: Experiment configuration
        output_dir: Directory to save results
        
    Returns:
        Dictionary with experiment results
    """
    print("Starting Black-box Resampling Experiment")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    print("Initializing components...")
    
    # Note: In a real implementation, you would load the actual model here
    # model = load_model(model_config)
    model = None  # Placeholder
    
    resampler = BlackboxResampler(
        model=model,
        num_rollouts=experiment_config.num_rollouts,
        temperature=experiment_config.resampling_temperature,
        top_p=experiment_config.resampling_top_p
    )
    
    analyzer = ImportanceAnalyzer(threshold=0.7)
    sentence_analyzer = SentenceAnalyzer(model=model, use_llm_labeling=False)
    
    # Run resampling
    print(f"Running resampling with {experiment_config.num_rollouts} rollouts...")
    resampling_results = resampler.resample_all_sentences(reasoning_trace)
    
    # Analyze importance
    print("Analyzing importance scores...")
    importance_analysis = analyzer.analyze_importance(resampling_results)
    
    # Analyze sentences
    print("Analyzing sentence categories...")
    labeled_sentences = sentence_analyzer.analyze_reasoning_trace(reasoning_trace)
    sentence_analysis = sentence_analyzer.get_analysis_summary(labeled_sentences)
    
    # Generate visualizations
    print("Generating visualizations...")
    importance_fig = analyzer.create_importance_plot(
        resampling_results, 
        save_path=os.path.join(output_dir, "importance_scores.png")
    )
    
    consistency_fig = analyzer.create_answer_consistency_plot(
        resampling_results,
        save_path=os.path.join(output_dir, "answer_consistency.png")
    )
    
    # Generate report
    print("Generating report...")
    report = analyzer.generate_report(resampling_results, importance_analysis)
    
    # Save results
    print("Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save resampling results
    resampler.save_results(
        resampling_results,
        os.path.join(output_dir, f"resampling_results_{timestamp}.json")
    )
    
    # Save sentence analysis
    sentence_analyzer.labeler.save_labeled_sentences(
        labeled_sentences,
        os.path.join(output_dir, f"labeled_sentences_{timestamp}.json")
    )
    
    # Save report
    with open(os.path.join(output_dir, f"report_{timestamp}.txt"), 'w') as f:
        f.write(report)
    
    # Compile results
    results = {
        "experiment_type": "blackbox_resampling",
        "timestamp": timestamp,
        "config": {
            "model": model_config.__dict__,
            "experiment": experiment_config.__dict__
        },
        "results": {
            "num_sentences": len(resampling_results),
            "thought_anchors": importance_analysis.thought_anchors,
            "top_important_sentences": analyzer.get_top_important_sentences(
                resampling_results, top_k=5
            ),
            "statistical_summary": importance_analysis.statistical_summary,
            "sentence_analysis": sentence_analysis
        },
        "files": {
            "resampling_results": f"resampling_results_{timestamp}.json",
            "labeled_sentences": f"labeled_sentences_{timestamp}.json",
            "report": f"report_{timestamp}.txt",
            "importance_plot": "importance_scores.png",
            "consistency_plot": "answer_consistency.png"
        }
    }
    
    # Save summary
    with open(os.path.join(output_dir, f"experiment_summary_{timestamp}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Experiment completed successfully!")
    print(f"Results saved to: {output_dir}")
    
    return results

def main():
    """Main function for running the experiment from command line."""
    parser = argparse.ArgumentParser(description="Run black-box resampling experiment")
    parser.add_argument("--reasoning_trace", type=str, required=True,
                       help="Path to file containing reasoning trace")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Directory to save results")
    parser.add_argument("--num_rollouts", type=int, default=100,
                       help="Number of rollouts per sentence")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    
    args = parser.parse_args()
    
    # Load reasoning trace
    with open(args.reasoning_trace, 'r') as f:
        reasoning_trace = f.read()
    
    # Create configurations
    model_config = ModelConfig()
    experiment_config = ExperimentConfig()
    experiment_config.num_rollouts = args.num_rollouts
    experiment_config.resampling_temperature = args.temperature
    experiment_config.resampling_top_p = args.top_p
    
    # Run experiment
    results = run_blackbox_experiment(
        reasoning_trace=reasoning_trace,
        model_config=model_config,
        experiment_config=experiment_config,
        output_dir=args.output_dir
    )
    
    print("Experiment completed!")
    print(f"Thought anchors found: {len(results['results']['thought_anchors'])}")

if __name__ == "__main__":
    main() 