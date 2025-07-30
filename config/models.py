"""
Model configuration for Thought Anchors experiments.
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Union
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for model settings."""
    
    # Model names
    reasoning_model_name: str = os.getenv("REASONING_MODEL_NAME", "deepseek-ai/deepseek-coder-33b-instruct")
    base_model_name: str = os.getenv("BASE_MODEL_NAME", "deepseek-ai/deepseek-coder-33b-instruct")
    
    # Model paths and caching
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "./models")
    
    # Hardware settings
    device: str = os.getenv("DEVICE", "cuda")
    mixed_precision: bool = os.getenv("MIXED_PRECISION", "true").lower() == "true"
    
    # Generation settings
    max_sequence_length: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "2048"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "4"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "4"))
    
    # API settings (for cloud models)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # Model-specific settings
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    
    # Attention analysis settings
    attention_layers: Union[str, List[int]] = os.getenv("ATTENTION_LAYERS", "all")
    receiver_head_threshold: float = float(os.getenv("RECEIVER_HEAD_THRESHOLD", "0.1"))
    suppression_mask_value: float = float(os.getenv("SUPPRESSION_MASK_VALUE", "-1e9"))
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.attention_layers == "all":
            self.attention_layers = "all"
        elif isinstance(self.attention_layers, str):
            self.attention_layers = [int(x.strip()) for x in self.attention_layers.split(",")]
    
    @property
    def is_local_model(self) -> bool:
        """Check if using a local model."""
        return not any(key in self.reasoning_model_name.lower() for key in ["gpt", "claude", "api"])
    
    @property
    def model_type(self) -> str:
        """Get the type of model being used."""
        if "gpt" in self.reasoning_model_name.lower():
            return "openai"
        elif "claude" in self.reasoning_model_name.lower():
            return "anthropic"
        else:
            return "local"

# Default configurations for different model types
DEFAULT_CONFIGS = {
    "deepseek": ModelConfig(
        reasoning_model_name="deepseek-ai/deepseek-coder-33b-instruct",
        base_model_name="deepseek-ai/deepseek-coder-33b-instruct",
    ),
    "llama": ModelConfig(
        reasoning_model_name="meta-llama/Llama-2-70b-chat-hf",
        base_model_name="meta-llama/Llama-2-70b-chat-hf",
    ),
    "qwen": ModelConfig(
        reasoning_model_name="Qwen/Qwen2.5-72B-Instruct",
        base_model_name="Qwen/Qwen2.5-72B-Instruct",
    ),
    "gpt4": ModelConfig(
        reasoning_model_name="gpt-4",
        base_model_name="gpt-4",
    ),
    "claude": ModelConfig(
        reasoning_model_name="claude-3-sonnet-20240229",
        base_model_name="claude-3-sonnet-20240229",
    ),
} 