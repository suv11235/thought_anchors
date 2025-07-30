"""
Model utility functions for Thought Anchors.
"""

import os
from typing import Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_config) -> Any:
    """
    Load a language model based on configuration.
    
    Args:
        model_config: Model configuration object
        
    Returns:
        Loaded model and tokenizer
    """
    if model_config.model_type == "local":
        return _load_local_model(model_config)
    elif model_config.model_type == "openai":
        return _setup_openai_model(model_config)
    elif model_config.model_type == "anthropic":
        return _setup_anthropic_model(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_config.model_type}")

def _load_local_model(model_config):
    """Load a local model using transformers."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.reasoning_model_name,
            cache_dir=model_config.model_cache_dir,
            trust_remote_code=model_config.trust_remote_code
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_config.reasoning_model_name,
            cache_dir=model_config.model_cache_dir,
            torch_dtype=getattr(torch, model_config.torch_dtype) if model_config.torch_dtype != "auto" else "auto",
            trust_remote_code=model_config.trust_remote_code,
            device_map="auto" if model_config.device == "cuda" else None
        )
        
        if model_config.device == "cuda" and not model_config.mixed_precision:
            model = model.half()
        
        return {"model": model, "tokenizer": tokenizer}
        
    except Exception as e:
        print(f"Error loading local model: {e}")
        return None

def _setup_openai_model(model_config):
    """Setup OpenAI API model."""
    try:
        import openai
        openai.api_key = model_config.openai_api_key
        
        return {
            "model": model_config.reasoning_model_name,
            "api_type": "openai",
            "client": openai
        }
    except ImportError:
        print("OpenAI library not installed. Install with: pip install openai")
        return None
    except Exception as e:
        print(f"Error setting up OpenAI model: {e}")
        return None

def _setup_anthropic_model(model_config):
    """Setup Anthropic API model."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=model_config.anthropic_api_key)
        
        return {
            "model": model_config.reasoning_model_name,
            "api_type": "anthropic",
            "client": client
        }
    except ImportError:
        print("Anthropic library not installed. Install with: pip install anthropic")
        return None
    except Exception as e:
        print(f"Error setting up Anthropic model: {e}")
        return None

def setup_model(model_config) -> Any:
    """
    Setup model for use in experiments.
    
    Args:
        model_config: Model configuration object
        
    Returns:
        Model object ready for use
    """
    model_data = load_model(model_config)
    
    if model_data is None:
        print("Warning: Could not load model. Using placeholder.")
        return None
    
    if "api_type" in model_data:
        # API model
        return model_data
    else:
        # Local model
        return model_data["model"]

def generate_text(model_data: Any, prompt: str, max_tokens: int = 200, 
                 temperature: float = 0.8, top_p: float = 0.9) -> str:
    """
    Generate text using the loaded model.
    
    Args:
        model_data: Model data from load_model
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Generated text
    """
    if model_data is None:
        return "ERROR: No model available"
    
    if "api_type" in model_data:
        # API model
        if model_data["api_type"] == "openai":
            return _generate_openai_text(model_data, prompt, max_tokens, temperature, top_p)
        elif model_data["api_type"] == "anthropic":
            return _generate_anthropic_text(model_data, prompt, max_tokens, temperature, top_p)
    else:
        # Local model
        return _generate_local_text(model_data, prompt, max_tokens, temperature, top_p)

def _generate_openai_text(model_data, prompt, max_tokens, temperature, top_p):
    """Generate text using OpenAI API."""
    try:
        response = model_data["client"].chat.completions.create(
            model=model_data["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: OpenAI generation failed - {e}"

def _generate_anthropic_text(model_data, prompt, max_tokens, temperature, top_p):
    """Generate text using Anthropic API."""
    try:
        response = model_data["client"].messages.create(
            model=model_data["model"],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"ERROR: Anthropic generation failed - {e}"

def _generate_local_text(model, prompt, max_tokens, temperature, top_p):
    """Generate text using local model."""
    try:
        # This is a simplified version - would need proper tokenization and generation
        return f"Generated text for: {prompt[:50]}..."
    except Exception as e:
        return f"ERROR: Local generation failed - {e}" 