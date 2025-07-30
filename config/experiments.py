"""
Experiment configuration for Thought Anchors experiments.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ExperimentConfig:
    """Configuration for experiment settings."""
    
    # Experiment basics
    experiment_name: str = "thought_anchors"
    seed: int = int(os.getenv("EXPERIMENT_SEED", "42"))
    
    # Data paths
    data_dir: str = os.getenv("DATA_DIR", "./data")
    raw_data_dir: str = os.getenv("RAW_DATA_DIR", "./data/raw")
    processed_data_dir: str = os.getenv("PROCESSED_DATA_DIR", "./data/processed")
    results_dir: str = os.getenv("RESULTS_DIR", "./data/results")
    
    # Logging and monitoring
    wandb_project: str = os.getenv("WANDB_PROJECT", "thought-anchors")
    wandb_entity: Optional[str] = os.getenv("WANDB_ENTITY")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Visualization settings
    plot_style: str = os.getenv("PLOT_STYLE", "seaborn")
    figure_format: str = os.getenv("FIGURE_FORMAT", "png")
    dpi: int = int(os.getenv("DPI", "300"))
    
    # Black-box resampling settings
    num_rollouts: int = int(os.getenv("NUM_ROLLOUTS", "100"))
    resampling_temperature: float = 0.8
    resampling_top_p: float = 0.9
    
    # Attention analysis settings
    min_sentence_length: int = 10
    max_sentence_length: int = 500
    attention_window_size: int = 50
    proximity_ignore_distance: int = 4
    
    # Sentence taxonomy settings
    taxonomy_categories: tuple = (
        "problem_setup",
        "plan_generation", 
        "fact_retrieval",
        "active_computation",
        "uncertainty_management",
        "result_consolidation",
        "self_checking",
        "final_answer_emission"
    )
    
    # Evaluation settings
    evaluation_metrics: tuple = (
        "accuracy",
        "f1_score",
        "precision",
        "recall",
        "kl_divergence"
    )
    
    def get_experiment_params(self, experiment_type: str) -> Dict[str, Any]:
        """Get parameters for a specific experiment type."""
        
        base_params = {
            "seed": self.seed,
            "data_dir": self.data_dir,
            "results_dir": self.results_dir,
        }
        
        if experiment_type == "blackbox_resampling":
            return {
                **base_params,
                "num_rollouts": self.num_rollouts,
                "temperature": self.resampling_temperature,
                "top_p": self.resampling_top_p,
            }
        
        elif experiment_type == "receiver_heads":
            return {
                **base_params,
                "min_sentence_length": self.min_sentence_length,
                "max_sentence_length": self.max_sentence_length,
                "attention_window_size": self.attention_window_size,
                "proximity_ignore_distance": self.proximity_ignore_distance,
            }
        
        elif experiment_type == "attention_suppression":
            return {
                **base_params,
                "min_sentence_length": self.min_sentence_length,
                "max_sentence_length": self.max_sentence_length,
            }
        
        else:
            return base_params

# Default experiment configurations
DEFAULT_EXPERIMENTS = {
    "quick_test": ExperimentConfig(
        experiment_name="quick_test",
        num_rollouts=10,
        attention_window_size=20,
    ),
    "full_analysis": ExperimentConfig(
        experiment_name="full_analysis",
        num_rollouts=100,
        attention_window_size=50,
    ),
    "paper_reproduction": ExperimentConfig(
        experiment_name="paper_reproduction",
        num_rollouts=100,
        attention_window_size=50,
        seed=42,
    ),
} 