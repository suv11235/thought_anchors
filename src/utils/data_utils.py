"""
Data utility functions for Thought Anchors.
"""

import os
import json
from typing import List, Dict, Any, Optional
import pandas as pd

def load_reasoning_traces(filepath: str) -> List[str]:
    """
    Load reasoning traces from a file.
    
    Args:
        filepath: Path to the file containing reasoning traces
        
    Returns:
        List of reasoning trace strings
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines or other separators if multiple traces
    traces = [trace.strip() for trace in content.split('\n\n') if trace.strip()]
    
    if not traces:
        # If no double newlines, treat entire content as one trace
        traces = [content.strip()]
    
    return traces

def save_results(results: Dict[str, Any], filepath: str):
    """
    Save experiment results to a file.
    
    Args:
        results: Results dictionary to save
        filepath: Path to save the results
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load experiment results from a file.
    
    Args:
        filepath: Path to the results file
        
    Returns:
        Loaded results dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_experiment_directory(base_dir: str, experiment_name: str) -> str:
    """
    Create a directory for experiment results.
    
    Args:
        base_dir: Base directory for results
        experiment_name: Name of the experiment
        
    Returns:
        Path to the created directory
    """
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def save_visualization(fig, filepath: str, dpi: int = 300):
    """
    Save a matplotlib figure.
    
    Args:
        fig: Matplotlib figure object
        filepath: Path to save the figure
        dpi: DPI for the saved figure
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')

def load_dataset_from_csv(filepath: str, text_column: str = "text") -> List[str]:
    """
    Load reasoning traces from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        text_column: Name of the column containing the text
        
    Returns:
        List of reasoning trace strings
    """
    df = pd.read_csv(filepath)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    return df[text_column].dropna().tolist()

def save_dataset_to_csv(data: List[str], filepath: str, text_column: str = "text"):
    """
    Save reasoning traces to a CSV file.
    
    Args:
        data: List of reasoning trace strings
        filepath: Path to save the CSV file
        text_column: Name of the column for the text
    """
    df = pd.DataFrame({text_column: data})
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def validate_reasoning_trace(text: str) -> bool:
    """
    Validate if a text looks like a reasoning trace.
    
    Args:
        text: Text to validate
        
    Returns:
        True if text appears to be a reasoning trace
    """
    if not text or len(text.strip()) < 50:
        return False
    
    # Check for common reasoning indicators
    reasoning_indicators = [
        "let me", "first", "then", "therefore", "thus", "so",
        "step by step", "calculate", "solve", "find", "determine",
        "because", "since", "as", "if", "then", "else"
    ]
    
    text_lower = text.lower()
    indicator_count = sum(1 for indicator in reasoning_indicators if indicator in text_lower)
    
    # Should have at least a few reasoning indicators
    return indicator_count >= 2

def clean_reasoning_trace(text: str) -> str:
    """
    Clean a reasoning trace text.
    
    Args:
        text: Raw reasoning trace text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove common artifacts
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text.strip()

def split_reasoning_traces(text: str, max_length: int = 2000) -> List[str]:
    """
    Split a long text into multiple reasoning traces.
    
    Args:
        text: Long text to split
        max_length: Maximum length per trace
        
    Returns:
        List of reasoning trace strings
    """
    if len(text) <= max_length:
        return [text]
    
    # Split by sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    traces = []
    current_trace = ""
    
    for sentence in sentences:
        if len(current_trace + sentence) <= max_length:
            current_trace += sentence + " "
        else:
            if current_trace:
                traces.append(current_trace.strip())
            current_trace = sentence + " "
    
    if current_trace:
        traces.append(current_trace.strip())
    
    return traces 