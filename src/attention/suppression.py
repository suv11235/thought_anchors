"""
Attention suppression for causal dependency analysis.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

@dataclass
class SuppressionResult:
    """Result of attention suppression analysis."""
    source_sentence: int
    target_sentences: List[int]
    kl_divergences: List[float]
    causal_effects: List[float]
    metadata: Dict[str, Any]

@dataclass
class CausalDependencyGraph:
    """Graph of causal dependencies between sentences."""
    nodes: List[int]
    edges: List[Tuple[int, int, float]]  # (source, target, effect_strength)
    adjacency_matrix: np.ndarray
    metadata: Dict[str, Any]

class AttentionSuppressionAnalyzer:
    """Analyzer for attention suppression and causal dependencies."""
    
    def __init__(self, model, mask_value: float = -1e9):
        """
        Initialize the attention suppression analyzer.
        
        Args:
            model: The language model to analyze
            mask_value: Value to use for masking attention
        """
        self.model = model
        self.mask_value = mask_value
    
    def analyze_sentence_dependencies(self, reasoning_trace: str) -> CausalDependencyGraph:
        """
        Analyze causal dependencies between sentences using attention suppression.
        
        Args:
            reasoning_trace: The reasoning trace to analyze
            
        Returns:
            CausalDependencyGraph with dependency information
        """
        # Split into sentences
        sentences = self._split_into_sentences(reasoning_trace)
        
        # Get token positions for each sentence
        sentence_token_ranges = self._get_sentence_token_ranges(reasoning_trace, sentences)
        
        # Analyze each sentence pair
        all_results = []
        for source_idx in range(len(sentences)):
            for target_idx in range(source_idx + 1, len(sentences)):
                result = self._analyze_sentence_pair(
                    reasoning_trace, sentences, sentence_token_ranges,
                    source_idx, target_idx
                )
                all_results.append(result)
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(sentences, all_results)
        
        return dependency_graph
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_sentence_token_ranges(self, text: str, sentences: List[str]) -> List[Tuple[int, int]]:
        """Get token position ranges for each sentence."""
        # This would use the model's tokenizer to get exact token positions
        # For now, use character-based approximation
        
        ranges = []
        current_pos = 0
        
        for sentence in sentences:
            start_pos = text.find(sentence, current_pos)
            if start_pos != -1:
                end_pos = start_pos + len(sentence)
                ranges.append((start_pos, end_pos))
                current_pos = end_pos
            else:
                # Fallback: estimate positions
                ranges.append((current_pos, current_pos + len(sentence)))
                current_pos += len(sentence) + 1
        
        return ranges
    
    def _analyze_sentence_pair(self, reasoning_trace: str, sentences: List[str],
                             sentence_token_ranges: List[Tuple[int, int]],
                             source_idx: int, target_idx: int) -> SuppressionResult:
        """Analyze the causal effect of one sentence on another."""
        
        # Get baseline logits for target sentence
        baseline_logits = self._get_sentence_logits(
            reasoning_trace, target_idx, sentence_token_ranges
        )
        
        # Suppress attention to source sentence and get modified logits
        suppressed_logits = self._get_suppressed_logits(
            reasoning_trace, source_idx, target_idx, sentence_token_ranges
        )
        
        # Calculate KL divergence
        kl_divergence = self._calculate_kl_divergence(baseline_logits, suppressed_logits)
        
        # Calculate causal effect (normalized KL divergence)
        causal_effect = kl_divergence / (np.sum(baseline_logits) + 1e-10)
        
        return SuppressionResult(
            source_sentence=source_idx,
            target_sentences=[target_idx],
            kl_divergences=[kl_divergence],
            causal_effects=[causal_effect],
            metadata={
                "baseline_logits": baseline_logits,
                "suppressed_logits": suppressed_logits
            }
        )
    
    def _get_sentence_logits(self, reasoning_trace: str, sentence_idx: int,
                           sentence_token_ranges: List[Tuple[int, int]]) -> np.ndarray:
        """Get logits for a specific sentence."""
        # This would use the model to get actual logits
        # For now, create dummy logits
        
        # Simulate logits for the sentence
        sentence_length = sentence_token_ranges[sentence_idx][1] - sentence_token_ranges[sentence_idx][0]
        vocab_size = 1000  # Assume vocabulary size
        
        # Create random logits (would be real logits in practice)
        logits = np.random.randn(sentence_length, vocab_size)
        
        return logits
    
    def _get_suppressed_logits(self, reasoning_trace: str, source_idx: int, target_idx: int,
                             sentence_token_ranges: List[Tuple[int, int]]) -> np.ndarray:
        """Get logits with attention to source sentence suppressed."""
        # This would modify the model's attention patterns and get new logits
        # For now, create modified dummy logits
        
        baseline_logits = self._get_sentence_logits(reasoning_trace, target_idx, sentence_token_ranges)
        
        # Simulate the effect of attention suppression
        # Add some noise to simulate the change
        noise_factor = 0.1
        suppressed_logits = baseline_logits + noise_factor * np.random.randn(*baseline_logits.shape)
        
        return suppressed_logits
    
    def _calculate_kl_divergence(self, p_logits: np.ndarray, q_logits: np.ndarray) -> float:
        """Calculate KL divergence between two logit distributions."""
        # Convert logits to probabilities
        p_probs = self._logits_to_probs(p_logits)
        q_probs = self._logits_to_probs(q_logits)
        
        # Calculate KL divergence
        kl_div = entropy(p_probs.flatten(), q_probs.flatten())
        
        return kl_div
    
    def _logits_to_probs(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities using softmax."""
        # Apply softmax along the vocabulary dimension
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        return probs
    
    def _build_dependency_graph(self, sentences: List[str], 
                              results: List[SuppressionResult]) -> CausalDependencyGraph:
        """Build a dependency graph from suppression results."""
        num_sentences = len(sentences)
        
        # Initialize adjacency matrix
        adjacency_matrix = np.zeros((num_sentences, num_sentences))
        
        # Add edges from results
        edges = []
        for result in results:
            source = result.source_sentence
            for target, effect in zip(result.target_sentences, result.causal_effects):
                adjacency_matrix[source, target] = effect
                edges.append((source, target, effect))
        
        return CausalDependencyGraph(
            nodes=list(range(num_sentences)),
            edges=edges,
            adjacency_matrix=adjacency_matrix,
            metadata={
                "num_sentences": num_sentences,
                "num_edges": len(edges),
                "total_effect": np.sum(adjacency_matrix)
            }
        )
    
    def identify_thought_anchors(self, dependency_graph: CausalDependencyGraph,
                               threshold: float = 0.1) -> List[int]:
        """Identify thought anchors based on causal dependencies."""
        # Calculate total outgoing effect for each sentence
        outgoing_effects = np.sum(dependency_graph.adjacency_matrix, axis=1)
        
        # Calculate total incoming effect for each sentence
        incoming_effects = np.sum(dependency_graph.adjacency_matrix, axis=0)
        
        # Combine effects (outgoing + incoming)
        total_effects = outgoing_effects + incoming_effects
        
        # Identify thought anchors
        thought_anchors = []
        for i, effect in enumerate(total_effects):
            if effect >= threshold:
                thought_anchors.append(i)
        
        return thought_anchors
    
    def get_dependency_paths(self, dependency_graph: CausalDependencyGraph,
                           source: int, target: int, max_length: int = 3) -> List[List[int]]:
        """Find dependency paths between two sentences."""
        # Simple path finding using adjacency matrix
        paths = []
        
        def find_paths(current: int, path: List[int], length: int):
            if length > max_length:
                return
            
            if current == target and length > 0:
                paths.append(path[:])
                return
            
            for next_node in range(len(dependency_graph.nodes)):
                if (dependency_graph.adjacency_matrix[current, next_node] > 0 and 
                    next_node not in path):
                    find_paths(next_node, path + [next_node], length + 1)
        
        find_paths(source, [source], 0)
        return paths
    
    def create_dependency_heatmap(self, dependency_graph: CausalDependencyGraph,
                                save_path: Optional[str] = None) -> plt.Figure:
        """Create a heatmap of causal dependencies."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(dependency_graph.adjacency_matrix, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Causal Effect Strength')
        
        # Add labels
        ax.set_xlabel('Target Sentence')
        ax.set_ylabel('Source Sentence')
        ax.set_title('Causal Dependencies Between Sentences')
        
        # Add sentence numbers
        ax.set_xticks(range(len(dependency_graph.nodes)))
        ax.set_yticks(range(len(dependency_graph.nodes)))
        ax.set_xticklabels([f'S{i}' for i in dependency_graph.nodes])
        ax.set_yticklabels([f'S{i}' for i in dependency_graph.nodes])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_dependency_network(self, dependency_graph: CausalDependencyGraph,
                                save_path: Optional[str] = None) -> plt.Figure:
        """Create a network visualization of dependencies."""
        import networkx as nx
        
        # Create network
        G = nx.DiGraph()
        
        # Add nodes
        for node in dependency_graph.nodes:
            G.add_node(node)
        
        # Add edges with weights
        for source, target, weight in dependency_graph.edges:
            if weight > 0.01:  # Only show significant dependencies
                G.add_edge(source, target, weight=weight)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=500, ax=ax)
        
        # Draw edges with varying thickness
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, 
                             edge_color='gray', alpha=0.7, ax=ax)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, ax=ax)
        
        ax.set_title('Causal Dependency Network')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_report(self, dependency_graph: CausalDependencyGraph,
                       thought_anchors: List[int]) -> str:
        """Generate a text report of the dependency analysis."""
        report = []
        report.append("=" * 60)
        report.append("CAUSAL DEPENDENCY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic statistics
        report.append("BASIC STATISTICS:")
        report.append(f"Total sentences: {dependency_graph.metadata['num_sentences']}")
        report.append(f"Total dependencies: {dependency_graph.metadata['num_edges']}")
        report.append(f"Thought anchors: {len(thought_anchors)}")
        report.append(f"Total causal effect: {dependency_graph.metadata['total_effect']:.3f}")
        report.append("")
        
        # Thought anchors
        if thought_anchors:
            report.append("THOUGHT ANCHORS:")
            for anchor in thought_anchors:
                outgoing = np.sum(dependency_graph.adjacency_matrix[anchor, :])
                incoming = np.sum(dependency_graph.adjacency_matrix[:, anchor])
                total_effect = outgoing + incoming
                report.append(f"  Sentence {anchor}: Total effect = {total_effect:.3f}")
            report.append("")
        
        # Strongest dependencies
        report.append("STRONGEST DEPENDENCIES:")
        sorted_edges = sorted(dependency_graph.edges, key=lambda x: x[2], reverse=True)
        for i, (source, target, effect) in enumerate(sorted_edges[:10]):
            report.append(f"  {i+1}. S{source} → S{target}: {effect:.3f}")
        report.append("")
        
        # Effect statistics
        effects = [edge[2] for edge in dependency_graph.edges]
        if effects:
            report.append("EFFECT STRENGTH STATISTICS:")
            report.append(f"  Mean: {np.mean(effects):.3f}")
            report.append(f"  Median: {np.median(effects):.3f}")
            report.append(f"  Std: {np.std(effects):.3f}")
            report.append(f"  Min: {np.min(effects):.3f}")
            report.append(f"  Max: {np.max(effects):.3f}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report) 