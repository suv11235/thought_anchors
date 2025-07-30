"""
Receiver heads analysis for attention patterns in reasoning traces.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ReceiverHeadResult:
    """Result of receiver head analysis."""
    sentence_positions: List[int]
    receiver_scores: List[float]
    broadcasting_sentences: List[int]
    attention_patterns: Dict[str, np.ndarray]
    metadata: Dict[str, Any]

class ReceiverHeadsAnalyzer:
    """Analyzer for receiver heads and broadcasting sentences."""
    
    def __init__(self, model, threshold: float = 0.1, 
                 proximity_ignore_distance: int = 4):
        """
        Initialize the receiver heads analyzer.
        
        Args:
            model: The language model to analyze
            threshold: Threshold for identifying receiver heads
            proximity_ignore_distance: Distance to ignore for proximity effects
        """
        self.model = model
        self.threshold = threshold
        self.proximity_ignore_distance = proximity_ignore_distance
    
    def analyze_receiver_heads(self, reasoning_trace: str) -> ReceiverHeadResult:
        """
        Analyze receiver heads in a reasoning trace.
        
        Args:
            reasoning_trace: The reasoning trace to analyze
            
        Returns:
            ReceiverHeadResult with analysis
        """
        # Tokenize the reasoning trace
        tokens = self._tokenize_text(reasoning_trace)
        
        # Get attention patterns
        attention_patterns = self._get_attention_patterns(tokens)
        
        # Identify receiver heads
        receiver_heads = self._identify_receiver_heads(attention_patterns)
        
        # Calculate receiver scores for sentences
        receiver_scores = self._calculate_receiver_scores(
            attention_patterns, receiver_heads, tokens
        )
        
        # Identify broadcasting sentences
        broadcasting_sentences = self._identify_broadcasting_sentences(
            receiver_scores, threshold=self.threshold
        )
        
        return ReceiverHeadResult(
            sentence_positions=list(range(len(receiver_scores))),
            receiver_scores=receiver_scores,
            broadcasting_sentences=broadcasting_sentences,
            attention_patterns=attention_patterns,
            metadata={
                "num_tokens": len(tokens),
                "num_sentences": len(receiver_scores),
                "receiver_heads": receiver_heads,
                "threshold": self.threshold
            }
        )
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into tokens."""
        # This would use the model's tokenizer
        if hasattr(self.model, 'tokenizer'):
            tokens = self.model.tokenizer.tokenize(text)
        else:
            # Simple word-based tokenization as fallback
            tokens = text.split()
        
        return tokens
    
    def _get_attention_patterns(self, tokens: List[str]) -> Dict[str, np.ndarray]:
        """Get attention patterns from the model."""
        # This is a placeholder - would be implemented based on the model
        # For now, create dummy attention patterns
        
        num_tokens = len(tokens)
        num_layers = 12  # Assume 12 layers
        num_heads = 12   # Assume 12 heads per layer
        
        attention_patterns = {}
        
        for layer in range(num_layers):
            for head in range(num_heads):
                key = f"layer_{layer}_head_{head}"
                
                # Create random attention pattern (would be real attention in practice)
                attention = np.random.rand(num_tokens, num_tokens)
                # Make it causal (lower triangular)
                attention = np.tril(attention)
                # Normalize
                attention = attention / attention.sum(axis=1, keepdims=True)
                
                attention_patterns[key] = attention
        
        return attention_patterns
    
    def _identify_receiver_heads(self, attention_patterns: Dict[str, np.ndarray]) -> List[str]:
        """Identify receiver heads based on attention patterns."""
        receiver_heads = []
        
        for head_key, attention in attention_patterns.items():
            # Calculate how focused the attention is
            # Receiver heads have more focused attention patterns
            
            # Calculate entropy of attention distributions
            entropies = []
            for i in range(attention.shape[0]):
                row = attention[i, :i+1]  # Only look at previous tokens
                if np.sum(row) > 0:
                    row = row / np.sum(row)
                    entropy = -np.sum(row * np.log(row + 1e-10))
                    entropies.append(entropy)
            
            if entropies:
                avg_entropy = np.mean(entropies)
                # Lower entropy means more focused attention
                if avg_entropy < 2.0:  # Threshold for focused attention
                    receiver_heads.append(head_key)
        
        return receiver_heads
    
    def _calculate_receiver_scores(self, attention_patterns: Dict[str, np.ndarray],
                                 receiver_heads: List[str], 
                                 tokens: List[str]) -> List[float]:
        """Calculate receiver scores for each sentence."""
        # Group tokens into sentences
        sentence_boundaries = self._find_sentence_boundaries(tokens)
        
        receiver_scores = []
        
        for i, (start, end) in enumerate(sentence_boundaries):
            # Calculate how much attention this sentence receives from receiver heads
            sentence_score = 0.0
            
            for head_key in receiver_heads:
                attention = attention_patterns[head_key]
                
                # Sum attention from all future tokens to this sentence
                sentence_attention = 0.0
                for future_pos in range(end, len(tokens)):
                    # Sum attention from this future token to tokens in this sentence
                    sentence_attention += np.sum(attention[future_pos, start:end])
                
                sentence_score += sentence_attention
            
            # Normalize by number of receiver heads
            if receiver_heads:
                sentence_score /= len(receiver_heads)
            
            receiver_scores.append(sentence_score)
        
        return receiver_scores
    
    def _find_sentence_boundaries(self, tokens: List[str]) -> List[Tuple[int, int]]:
        """Find sentence boundaries in tokens."""
        boundaries = []
        start = 0
        
        for i, token in enumerate(tokens):
            # Simple heuristic: sentence ends with period, exclamation, or question mark
            if token in ['.', '!', '?']:
                boundaries.append((start, i + 1))
                start = i + 1
        
        # Add final sentence if not empty
        if start < len(tokens):
            boundaries.append((start, len(tokens)))
        
        return boundaries
    
    def _identify_broadcasting_sentences(self, receiver_scores: List[float],
                                       threshold: float) -> List[int]:
        """Identify broadcasting sentences based on receiver scores."""
        broadcasting_sentences = []
        
        for i, score in enumerate(receiver_scores):
            if score >= threshold:
                broadcasting_sentences.append(i)
        
        return broadcasting_sentences
    
    def compare_with_base_model(self, reasoning_trace: str, 
                              base_model) -> Dict[str, Any]:
        """Compare receiver heads with a base model."""
        # Analyze reasoning model
        reasoning_result = self.analyze_receiver_heads(reasoning_trace)
        
        # Analyze base model
        base_analyzer = ReceiverHeadsAnalyzer(base_model, self.threshold)
        base_result = base_analyzer.analyze_receiver_heads(reasoning_trace)
        
        # Compare results
        comparison = {
            "reasoning_model": {
                "num_receiver_heads": len(reasoning_result.metadata["receiver_heads"]),
                "num_broadcasting_sentences": len(reasoning_result.broadcasting_sentences),
                "avg_receiver_score": np.mean(reasoning_result.receiver_scores)
            },
            "base_model": {
                "num_receiver_heads": len(base_result.metadata["receiver_heads"]),
                "num_broadcasting_sentences": len(base_result.broadcasting_sentences),
                "avg_receiver_score": np.mean(base_result.receiver_scores)
            },
            "differences": {
                "receiver_heads_diff": len(reasoning_result.metadata["receiver_heads"]) - 
                                     len(base_result.metadata["receiver_heads"]),
                "broadcasting_sentences_diff": len(reasoning_result.broadcasting_sentences) - 
                                            len(base_result.broadcasting_sentences),
                "avg_score_diff": np.mean(reasoning_result.receiver_scores) - 
                                np.mean(base_result.receiver_scores)
            }
        }
        
        return comparison
    
    def create_attention_heatmap(self, result: ReceiverHeadResult,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create a heatmap of attention patterns."""
        # Use the first receiver head for visualization
        if result.metadata["receiver_heads"]:
            head_key = result.metadata["receiver_heads"][0]
            attention = result.attention_patterns[head_key]
        else:
            # Use a random head if no receiver heads found
            head_key = list(result.attention_patterns.keys())[0]
            attention = result.attention_patterns[head_key]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(attention, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight')
        
        # Add labels
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Token Position')
        ax.set_title(f'Attention Pattern: {head_key}')
        
        # Highlight broadcasting sentences
        for sentence_pos in result.broadcasting_sentences:
            # This would need sentence boundaries to highlight properly
            pass
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_receiver_scores_plot(self, result: ReceiverHeadResult,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Create a plot of receiver scores by sentence position."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Receiver scores by position
        positions = result.sentence_positions
        scores = result.receiver_scores
        
        ax1.scatter(positions, scores, alpha=0.6, s=50)
        ax1.axhline(y=self.threshold, color='r', linestyle='--', 
                   label=f'Broadcasting Threshold ({self.threshold})')
        
        # Highlight broadcasting sentences
        for pos in result.broadcasting_sentences:
            ax1.scatter(pos, scores[pos], color='red', s=100, alpha=0.8, 
                       marker='o', edgecolors='black')
        
        ax1.set_xlabel('Sentence Position')
        ax1.set_ylabel('Receiver Score')
        ax1.set_title('Receiver Scores by Sentence Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distribution of receiver scores
        ax2.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(x=self.threshold, color='r', linestyle='--', 
                   label=f'Broadcasting Threshold ({self.threshold})')
        ax2.set_xlabel('Receiver Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Receiver Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_report(self, result: ReceiverHeadResult) -> str:
        """Generate a text report of the receiver heads analysis."""
        report = []
        report.append("=" * 60)
        report.append("RECEIVER HEADS ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic statistics
        report.append("BASIC STATISTICS:")
        report.append(f"Total sentences analyzed: {len(result.sentence_positions)}")
        report.append(f"Receiver heads identified: {len(result.metadata['receiver_heads'])}")
        report.append(f"Broadcasting sentences: {len(result.broadcasting_sentences)}")
        report.append(f"Broadcasting ratio: {len(result.broadcasting_sentences)/len(result.sentence_positions):.2%}")
        report.append("")
        
        # Receiver heads details
        report.append("RECEIVER HEADS:")
        for head in result.metadata["receiver_heads"]:
            report.append(f"  {head}")
        report.append("")
        
        # Broadcasting sentences
        if result.broadcasting_sentences:
            report.append("BROADCASTING SENTENCES:")
            for pos in result.broadcasting_sentences:
                score = result.receiver_scores[pos]
                report.append(f"  Position {pos} (Score: {score:.3f})")
            report.append("")
        
        # Score statistics
        scores = np.array(result.receiver_scores)
        report.append("RECEIVER SCORE STATISTICS:")
        report.append(f"  Mean: {np.mean(scores):.3f}")
        report.append(f"  Median: {np.median(scores):.3f}")
        report.append(f"  Std: {np.std(scores):.3f}")
        report.append(f"  Min: {np.min(scores):.3f}")
        report.append(f"  Max: {np.max(scores):.3f}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report) 