"""
Receiver heads experiment for Thought Anchors.
"""

import os
import json
import argparse
from typing import List, Dict, Any
from datetime import datetime

from src.attention import ReceiverHeadsAnalyzer
from src.sentence_analysis import SentenceAnalyzer
from config.models import ModelConfig
from config.experiments import ExperimentConfig

def run_receiver_heads_experiment(
    reasoning_trace: str,
    model_config: ModelConfig,
    experiment_config: ExperimentConfig,
    output_dir: str = "./results"
) -> Dict[str, Any]:
    """
    Run receiver heads experiment.
    
    Args:
        reasoning_trace: The reasoning trace to analyze
        model_config: Model configuration
        experiment_config: Experiment configuration
        output_dir: Directory to save results
        
    Returns:
        Dictionary with experiment results
    """
    print("Starting Receiver Heads Experiment")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    print("Initializing components...")
    
    # Note: In a real implementation, you would load the actual model here
    # model = load_model(model_config)
    model = None  # Placeholder
    
    analyzer = ReceiverHeadsAnalyzer(
        model=model,
        threshold=model_config.receiver_head_threshold,
        proximity_ignore_distance=experiment_config.proximity_ignore_distance
    )
    
    sentence_analyzer = SentenceAnalyzer(model=model, use_llm_labeling=False)
    
    # Run receiver heads analysis
    print("Analyzing receiver heads...")
    receiver_result = analyzer.analyze_receiver_heads(reasoning_trace)
    
    # Analyze sentences
    print("Analyzing sentence categories...")
    labeled_sentences = sentence_analyzer.analyze_reasoning_trace(reasoning_trace)
    sentence_analysis = sentence_analyzer.get_analysis_summary(labeled_sentences)
    
    # Generate visualizations
    print("Generating visualizations...")
    attention_fig = analyzer.create_attention_heatmap(
        receiver_result,
        save_path=os.path.join(output_dir, "attention_heatmap.png")
    )
    
    scores_fig = analyzer.create_receiver_scores_plot(
        receiver_result,
        save_path=os.path.join(output_dir, "receiver_scores.png")
    )
    
    # Generate report
    print("Generating report...")
    report = analyzer.generate_report(receiver_result)
    
    # Save results
    print("Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save sentence analysis
    sentence_analyzer.labeler.save_labeled_sentences(
        labeled_sentences,
        os.path.join(output_dir, f"labeled_sentences_{timestamp}.json")
    )
    
    # Save receiver heads data
    receiver_data = {
        "sentence_positions": receiver_result.sentence_positions,
        "receiver_scores": receiver_result.receiver_scores,
        "broadcasting_sentences": receiver_result.broadcasting_sentences,
        "metadata": receiver_result.metadata
    }
    
    with open(os.path.join(output_dir, f"receiver_heads_data_{timestamp}.json"), 'w') as f:
        json.dump(receiver_data, f, indent=2)
    
    # Save report
    with open(os.path.join(output_dir, f"report_{timestamp}.txt"), 'w') as f:
        f.write(report)
    
    # Compile results
    results = {
        "experiment_type": "receiver_heads",
        "timestamp": timestamp,
        "config": {
            "model": model_config.__dict__,
            "experiment": experiment_config.__dict__
        },
        "results": {
            "num_sentences": len(receiver_result.sentence_positions),
            "receiver_heads": receiver_result.metadata["receiver_heads"],
            "broadcasting_sentences": receiver_result.broadcasting_sentences,
            "receiver_scores": receiver_result.receiver_scores,
            "sentence_analysis": sentence_analysis
        },
        "files": {
            "labeled_sentences": f"labeled_sentences_{timestamp}.json",
            "receiver_heads_data": f"receiver_heads_data_{timestamp}.json",
            "report": f"report_{timestamp}.txt",
            "attention_heatmap": "attention_heatmap.png",
            "receiver_scores_plot": "receiver_scores.png"
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
    parser = argparse.ArgumentParser(description="Run receiver heads experiment")
    parser.add_argument("--reasoning_trace", type=str, required=True,
                       help="Path to file containing reasoning trace")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="Threshold for identifying receiver heads")
    
    args = parser.parse_args()
    
    # Load reasoning trace
    with open(args.reasoning_trace, 'r') as f:
        reasoning_trace = f.read()
    
    # Create configurations
    model_config = ModelConfig()
    model_config.receiver_head_threshold = args.threshold
    
    experiment_config = ExperimentConfig()
    
    # Run experiment
    results = run_receiver_heads_experiment(
        reasoning_trace=reasoning_trace,
        model_config=model_config,
        experiment_config=experiment_config,
        output_dir=args.output_dir
    )
    
    print("Experiment completed!")
    print(f"Broadcasting sentences found: {len(results['results']['broadcasting_sentences'])}")

if __name__ == "__main__":
    main() 