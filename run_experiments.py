#!/usr/bin/env python3
"""
Main script to run Thought Anchors experiments.
"""

import os
import json
import argparse
from typing import Dict, Any, List
from datetime import datetime

from experiments.blackbox_resampling import run_blackbox_experiment
from experiments.receiver_heads import run_receiver_heads_experiment
from experiments.attention_suppression import run_attention_suppression_experiment
from config.models import ModelConfig, DEFAULT_CONFIGS
from config.experiments import ExperimentConfig, DEFAULT_EXPERIMENTS

def load_reasoning_trace(filepath: str) -> str:
    """Load reasoning trace from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def run_all_experiments(
    reasoning_trace: str,
    model_config: ModelConfig,
    experiment_config: ExperimentConfig,
    output_dir: str = "./results",
    methods: List[str] = None
) -> Dict[str, Any]:
    """
    Run all Thought Anchors experiments.
    
    Args:
        reasoning_trace: The reasoning trace to analyze
        model_config: Model configuration
        experiment_config: Experiment configuration
        output_dir: Directory to save results
        methods: List of methods to run (default: all)
        
    Returns:
        Dictionary with all experiment results
    """
    if methods is None:
        methods = ["blackbox", "receiver_heads", "attention_suppression"]
    
    print("Starting Thought Anchors Experiments")
    print("=" * 50)
    print(f"Methods to run: {methods}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run black-box resampling
    if "blackbox" in methods:
        print("Running Black-box Resampling Experiment...")
        blackbox_results = run_blackbox_experiment(
            reasoning_trace=reasoning_trace,
            model_config=model_config,
            experiment_config=experiment_config,
            output_dir=os.path.join(output_dir, "blackbox")
        )
        results["blackbox"] = blackbox_results
    
    # Run receiver heads analysis
    if "receiver_heads" in methods:
        print("Running Receiver Heads Experiment...")
        receiver_results = run_receiver_heads_experiment(
            reasoning_trace=reasoning_trace,
            model_config=model_config,
            experiment_config=experiment_config,
            output_dir=os.path.join(output_dir, "receiver_heads")
        )
        results["receiver_heads"] = receiver_results
    
    # Run attention suppression analysis
    if "attention_suppression" in methods:
        print("Running Attention Suppression Experiment...")
        suppression_results = run_attention_suppression_experiment(
            reasoning_trace=reasoning_trace,
            model_config=model_config,
            experiment_config=experiment_config,
            output_dir=os.path.join(output_dir, "attention_suppression")
        )
        results["attention_suppression"] = suppression_results
    
    # Generate combined report
    print("Generating combined report...")
    combined_report = generate_combined_report(results, reasoning_trace)
    
    # Save combined results
    combined_results = {
        "experiment_type": "thought_anchors_combined",
        "timestamp": timestamp,
        "methods_run": methods,
        "config": {
            "model": model_config.__dict__,
            "experiment": experiment_config.__dict__
        },
        "results": results,
        "combined_report": combined_report
    }
    
    with open(os.path.join(output_dir, f"combined_results_{timestamp}.json"), 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    # Save combined report
    with open(os.path.join(output_dir, f"combined_report_{timestamp}.txt"), 'w') as f:
        f.write(combined_report)
    
    print("All experiments completed successfully!")
    print(f"Results saved to: {output_dir}")
    
    return combined_results

def generate_combined_report(results: Dict[str, Any], reasoning_trace: str) -> str:
    """Generate a combined report from all experiment results."""
    report = []
    report.append("=" * 80)
    report.append("THOUGHT ANCHORS - COMBINED EXPERIMENT REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary
    report.append("EXPERIMENT SUMMARY:")
    report.append(f"Reasoning trace length: {len(reasoning_trace)} characters")
    report.append(f"Methods run: {', '.join(results.keys())}")
    report.append("")
    
    # Black-box results
    if "blackbox" in results:
        report.append("BLACK-BOX RESAMPLING RESULTS:")
        blackbox = results["blackbox"]["results"]
        report.append(f"  Total sentences: {blackbox['num_sentences']}")
        report.append(f"  Thought anchors: {len(blackbox['thought_anchors'])}")
        report.append(f"  Top important sentence: {blackbox['top_important_sentences'][0] if blackbox['top_important_sentences'] else 'None'}")
        report.append("")
    
    # Receiver heads results
    if "receiver_heads" in results:
        report.append("RECEIVER HEADS RESULTS:")
        receiver = results["receiver_heads"]["results"]
        report.append(f"  Total sentences: {receiver['num_sentences']}")
        report.append(f"  Broadcasting sentences: {len(receiver['broadcasting_sentences'])}")
        report.append(f"  Receiver heads: {len(receiver['receiver_heads'])}")
        report.append("")
    
    # Attention suppression results
    if "attention_suppression" in results:
        report.append("ATTENTION SUPPRESSION RESULTS:")
        suppression = results["attention_suppression"]["results"]
        report.append(f"  Total sentences: {suppression['num_sentences']}")
        report.append(f"  Total dependencies: {suppression['num_edges']}")
        report.append(f"  Thought anchors: {len(suppression['thought_anchors'])}")
        report.append("")
    
    # Cross-method analysis
    report.append("CROSS-METHOD ANALYSIS:")
    
    # Find common thought anchors across methods
    thought_anchors_by_method = {}
    if "blackbox" in results:
        thought_anchors_by_method["blackbox"] = set(results["blackbox"]["results"]["thought_anchors"])
    if "receiver_heads" in results:
        thought_anchors_by_method["receiver_heads"] = set(results["receiver_heads"]["results"]["broadcasting_sentences"])
    if "attention_suppression" in results:
        thought_anchors_by_method["attention_suppression"] = set(results["attention_suppression"]["results"]["thought_anchors"])
    
    if len(thought_anchors_by_method) > 1:
        # Find intersection
        common_anchors = set.intersection(*thought_anchors_by_method.values())
        report.append(f"  Common thought anchors across all methods: {len(common_anchors)}")
        if common_anchors:
            report.append(f"    Positions: {sorted(common_anchors)}")
        
        # Find union
        all_anchors = set.union(*thought_anchors_by_method.values())
        report.append(f"  Total unique thought anchors: {len(all_anchors)}")
        report.append(f"    Positions: {sorted(all_anchors)}")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    """Main function for running experiments from command line."""
    parser = argparse.ArgumentParser(description="Run Thought Anchors experiments")
    parser.add_argument("--reasoning_trace", type=str, required=True,
                       help="Path to file containing reasoning trace")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Directory to save results")
    parser.add_argument("--methods", type=str, nargs="+", 
                       choices=["blackbox", "receiver_heads", "attention_suppression"],
                       default=["blackbox", "receiver_heads", "attention_suppression"],
                       help="Methods to run")
    parser.add_argument("--model", type=str, default="deepseek",
                       choices=list(DEFAULT_CONFIGS.keys()),
                       help="Model configuration to use")
    parser.add_argument("--experiment", type=str, default="full_analysis",
                       choices=list(DEFAULT_EXPERIMENTS.keys()),
                       help="Experiment configuration to use")
    
    args = parser.parse_args()
    
    # Load reasoning trace
    reasoning_trace = load_reasoning_trace(args.reasoning_trace)
    
    # Get configurations
    model_config = DEFAULT_CONFIGS[args.model]
    experiment_config = DEFAULT_EXPERIMENTS[args.experiment]
    
    # Run experiments
    results = run_all_experiments(
        reasoning_trace=reasoning_trace,
        model_config=model_config,
        experiment_config=experiment_config,
        output_dir=args.output_dir,
        methods=args.methods
    )
    
    print("All experiments completed!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 