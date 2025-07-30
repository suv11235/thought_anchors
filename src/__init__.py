"""
Thought Anchors: Analysis of LLM reasoning steps through sentence-level attribution methods.

This package implements the three main methods from the paper:
1. Black-box resampling for counterfactual importance
2. Receiver heads analysis for attention patterns
3. Attention suppression for causal dependencies
"""

__version__ = "0.1.0"
__author__ = "Thought Anchors Contributors"

from .sentence_analysis import SentenceAnalyzer
from .blackbox import BlackboxResampler
from .attention import AttentionAnalyzer

__all__ = [
    "SentenceAnalyzer",
    "BlackboxResampler", 
    "AttentionAnalyzer",
] 