"""
Attention suppression experiment for Thought Anchors.
"""

import os
import json
import argparse
from typing import List, Dict, Any
from datetime import datetime

from src.attention import AttentionSuppressionAnalyzer
from src.sentence_analysis import SentenceAnalyzer
from config.models import ModelConfig
from config.experiments import ExperimentConfig

def run_attention_suppression_experiment(
    reasoning_trace: str,
    model_config: ModelConfig,
    experiment_config: ExperimentConfig,
    output_dir: str = "./results"
) -> Dict[str, Any]:
    """
    Run attention suppression experiment.
    
    Args:
        reasoning_trace: The reasoning trace to analyze
        model_config: Model configuration
        experiment_config: Experiment configuration
        output_dir: Directory to save results
        
    Returns:
        Dictionary with experiment results
    """
    print("Starting Attention Suppression Experiment")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    print("Initializing components...")
    
    # Note: In a real implementation, you would load the actual model here
    # model = load_model(model_config)
    model = None  # Placeholder
    
    analyzer = AttentionSuppressionAnalyzer(
        model=model,
        mask_value=model_config.suppression_mask_value
    )
    
    sentence_analyzer = SentenceAnalyzer(model=model, use_llm_labeling=False)
    
    # Run attention suppression analysis
    print("Analyzing causal dependencies...")
    dependency_graph = analyzer.analyze_sentence_dependencies(reasoning_trace)
    
    # Identify thought anchors
    print("Identifying thought anchors...")
    thought_anchors = analyzer.identify_thought_anchors(dependency_graph, threshold=0.1)
    
    # Analyze sentences
    print("Analyzing sentence categories...")
    labeled_sentences = sentence_analyzer.analyze_reasoning_trace(reasoning_trace)
    sentence_analysis = sentence_analyzer.get_analysis_summary(labeled_sentences)
    
    # Generate visualizations
    print("Generating visualizations...")
    heatmap_fig = analyzer.create_dependency_heatmap(
        dependency_graph,
        save_path=os.path.join(output_dir, "dependency_heatmap.png")
    )
    
    network_fig = analyzer.create_dependency_network(
        dependency_graph,
        save_path=os.path.join(output_dir, "dependency_network.png")
    )
    
    # Generate report
    print("Generating report...")
    report = analyzer.generate_report(dependency_graph, thought_anchors)
    
    # Save results
    print("Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save sentence analysis
    sentence_analyzer.labeler.save_labeled_sentences(
        labeled_sentences,
        os.path.join(output_dir, f"labeled_sentences_{timestamp}.json")
    )
    
    # Save dependency graph data
    dependency_data = {
        "nodes": dependency_graph.nodes,
        "edges": dependency_graph.edges,
        "adjacency_matrix": dependency_graph.adjacency_matrix.tolist(),
        "metadata": dependency_graph.metadata,
        "thought_anchors": thought_anchors
    }
    
    with open(os.path.join(output_dir, f"dependency_graph_{timestamp}.json"), 'w') as f:
        json.dump(dependency_data, f, indent=2)
    
    # Save report
    with open(os.path.join(output_dir, f"report_{timestamp}.txt"), 'w') as f:
        f.write(report)
    
    # Compile results
    results = {
        "experiment_type": "attention_suppression",
        "timestamp": timestamp,
        "config": {
            "model": model_config.__dict__,
            "experiment": experiment_config.__dict__
        },
        "results": {
            "num_sentences": dependency_graph.metadata["num_sentences"],
            "num_edges": dependency_graph.metadata["num_edges"],
            "total_effect": dependency_graph.metadata["total_effect"],
            "thought_anchors": thought_anchors,
            "sentence_analysis": sentence_analysis
        },
        "files": {
            "labeled_sentences": f"labeled_sentences_{timestamp}.json",
            "dependency_graph": f"dependency_graph_{timestamp}.json",
            "report": f"report_{timestamp}.txt",
            "dependency_heatmap": "dependency_heatmap.png",
            "dependency_network": "dependency_network.png"
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
    parser = argparse.ArgumentParser(description="Run attention suppression experiment")
    parser.add_argument("--reasoning_trace", type=str, required=True,
                       help="Path to file containing reasoning trace")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Directory to save results")
    parser.add_argument("--mask_value", type=float, default=-1e9,
                       help="Value to use for masking attention")
    
    args = parser.parse_args()
    
    # Load reasoning trace
    with open(args.reasoning_trace, 'r') as f:
        reasoning_trace = f.read()
    
    # Create configurations
    model_config = ModelConfig()
    model_config.suppression_mask_value = args.mask_value
    
    experiment_config = ExperimentConfig()
    
    # Run experiment
    results = run_attention_suppression_experiment(
        reasoning_trace=reasoning_trace,
        model_config=model_config,
        experiment_config=experiment_config,
        output_dir=args.output_dir
    )
    
    print("Experiment completed!")
    print(f"Thought anchors found: {len(results['results']['thought_anchors'])}")

if __name__ == "__main__":
    main() 