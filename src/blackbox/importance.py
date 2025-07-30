"""
Importance analysis for black-box resampling results.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

from .resampling import ResamplingResult

@dataclass
class ImportanceAnalysis:
    """Analysis of sentence importance from resampling results."""
    sentence_positions: List[int]
    importance_scores: List[float]
    thought_anchors: List[int]
    category_analysis: Dict[str, Any]
    statistical_summary: Dict[str, float]

class ImportanceAnalyzer:
    """Analyzer for importance scores from black-box resampling."""
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize the importance analyzer.
        
        Args:
            threshold: Threshold for identifying thought anchors
        """
        self.threshold = threshold
    
    def analyze_importance(self, results: List[ResamplingResult]) -> ImportanceAnalysis:
        """
        Analyze importance scores from resampling results.
        
        Args:
            results: List of ResamplingResult objects
            
        Returns:
            ImportanceAnalysis with comprehensive analysis
        """
        # Extract basic data
        sentence_positions = [r.metadata["sentence_position"] for r in results]
        importance_scores = [r.importance_score for r in results]
        
        # Identify thought anchors
        thought_anchors = [
            pos for pos, score in zip(sentence_positions, importance_scores)
            if score >= self.threshold
        ]
        
        # Category analysis (if available)
        category_analysis = self._analyze_categories(results)
        
        # Statistical summary
        statistical_summary = self._calculate_statistics(importance_scores)
        
        return ImportanceAnalysis(
            sentence_positions=sentence_positions,
            importance_scores=importance_scores,
            thought_anchors=thought_anchors,
            category_analysis=category_analysis,
            statistical_summary=statistical_summary
        )
    
    def _analyze_categories(self, results: List[ResamplingResult]) -> Dict[str, Any]:
        """Analyze importance by sentence categories."""
        # This would require sentence categorization
        # For now, return basic analysis
        return {
            "total_sentences": len(results),
            "thought_anchor_ratio": len([r for r in results if r.importance_score >= self.threshold]) / len(results),
            "average_importance": np.mean([r.importance_score for r in results]),
        }
    
    def _calculate_statistics(self, importance_scores: List[float]) -> Dict[str, float]:
        """Calculate statistical summary of importance scores."""
        scores = np.array(importance_scores)
        
        return {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "q25": float(np.percentile(scores, 25)),
            "q75": float(np.percentile(scores, 75)),
        }
    
    def get_top_important_sentences(self, results: List[ResamplingResult], 
                                  top_k: int = 10) -> List[Tuple[int, float, str]]:
        """Get the top-k most important sentences."""
        # Sort by importance score
        sorted_results = sorted(results, key=lambda x: x.importance_score, reverse=True)
        
        top_sentences = []
        for i, result in enumerate(sorted_results[:top_k]):
            top_sentences.append((
                result.metadata["sentence_position"],
                result.importance_score,
                result.original_sentence
            ))
        
        return top_sentences
    
    def get_thought_anchors(self, results: List[ResamplingResult]) -> List[Tuple[int, float, str]]:
        """Get all thought anchors (sentences above threshold)."""
        thought_anchors = []
        
        for result in results:
            if result.importance_score >= self.threshold:
                thought_anchors.append((
                    result.metadata["sentence_position"],
                    result.importance_score,
                    result.original_sentence
                ))
        
        # Sort by importance score
        thought_anchors.sort(key=lambda x: x[1], reverse=True)
        return thought_anchors
    
    def analyze_answer_consistency(self, results: List[ResamplingResult]) -> Dict[str, Any]:
        """Analyze the consistency of answers across resampling."""
        consistency_analysis = {
            "high_consistency": [],  # Sentences that don't change answers much
            "low_consistency": [],   # Sentences that change answers a lot
            "answer_diversity": {}   # Distribution of answer changes
        }
        
        for result in results:
            # Calculate answer diversity
            unique_answers = len(result.answer_distribution)
            total_answers = sum(result.answer_distribution.values())
            
            if total_answers > 0:
                diversity_ratio = unique_answers / total_answers
                
                if diversity_ratio < 0.3:  # High consistency
                    consistency_analysis["high_consistency"].append({
                        "position": result.metadata["sentence_position"],
                        "diversity_ratio": diversity_ratio,
                        "importance_score": result.importance_score
                    })
                elif diversity_ratio > 0.7:  # Low consistency
                    consistency_analysis["low_consistency"].append({
                        "position": result.metadata["sentence_position"],
                        "diversity_ratio": diversity_ratio,
                        "importance_score": result.importance_score
                    })
                
                consistency_analysis["answer_diversity"][result.metadata["sentence_position"]] = {
                    "diversity_ratio": diversity_ratio,
                    "unique_answers": unique_answers,
                    "total_answers": total_answers
                }
        
        return consistency_analysis
    
    def create_importance_plot(self, results: List[ResamplingResult], 
                             save_path: Optional[str] = None) -> plt.Figure:
        """Create a plot showing importance scores by sentence position."""
        positions = [r.metadata["sentence_position"] for r in results]
        scores = [r.importance_score for r in results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Importance scores by position
        ax1.scatter(positions, scores, alpha=0.6, s=50)
        ax1.axhline(y=self.threshold, color='r', linestyle='--', 
                   label=f'Thought Anchor Threshold ({self.threshold})')
        ax1.set_xlabel('Sentence Position')
        ax1.set_ylabel('Importance Score')
        ax1.set_title('Sentence Importance Scores by Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distribution of importance scores
        ax2.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(x=self.threshold, color='r', linestyle='--', 
                   label=f'Thought Anchor Threshold ({self.threshold})')
        ax2.set_xlabel('Importance Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Importance Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_answer_consistency_plot(self, results: List[ResamplingResult],
                                     save_path: Optional[str] = None) -> plt.Figure:
        """Create a plot showing answer consistency analysis."""
        positions = []
        diversity_ratios = []
        importance_scores = []
        
        for result in results:
            unique_answers = len(result.answer_distribution)
            total_answers = sum(result.answer_distribution.values())
            
            if total_answers > 0:
                diversity_ratio = unique_answers / total_answers
                positions.append(result.metadata["sentence_position"])
                diversity_ratios.append(diversity_ratio)
                importance_scores.append(result.importance_score)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Answer diversity vs position
        scatter = ax1.scatter(positions, diversity_ratios, c=importance_scores, 
                            cmap='viridis', s=50, alpha=0.7)
        ax1.set_xlabel('Sentence Position')
        ax1.set_ylabel('Answer Diversity Ratio')
        ax1.set_title('Answer Diversity by Sentence Position')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Importance Score')
        
        # Plot 2: Answer diversity vs importance score
        ax2.scatter(importance_scores, diversity_ratios, alpha=0.6, s=50)
        ax2.set_xlabel('Importance Score')
        ax2.set_ylabel('Answer Diversity Ratio')
        ax2.set_title('Answer Diversity vs Importance Score')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_report(self, results: List[ResamplingResult], 
                       analysis: ImportanceAnalysis) -> str:
        """Generate a text report of the importance analysis."""
        report = []
        report.append("=" * 60)
        report.append("THOUGHT ANCHORS IMPORTANCE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic statistics
        report.append("BASIC STATISTICS:")
        report.append(f"Total sentences analyzed: {len(results)}")
        report.append(f"Thought anchors identified: {len(analysis.thought_anchors)}")
        report.append(f"Thought anchor ratio: {len(analysis.thought_anchors)/len(results):.2%}")
        report.append("")
        
        # Statistical summary
        report.append("IMPORTANCE SCORE STATISTICS:")
        for stat, value in analysis.statistical_summary.items():
            report.append(f"  {stat.capitalize()}: {value:.3f}")
        report.append("")
        
        # Top important sentences
        top_sentences = self.get_top_important_sentences(results, top_k=5)
        report.append("TOP 5 MOST IMPORTANT SENTENCES:")
        for i, (pos, score, sentence) in enumerate(top_sentences, 1):
            report.append(f"  {i}. Position {pos} (Score: {score:.3f})")
            report.append(f"     Text: {sentence[:100]}{'...' if len(sentence) > 100 else ''}")
            report.append("")
        
        # Thought anchors
        thought_anchors = self.get_thought_anchors(results)
        if thought_anchors:
            report.append("THOUGHT ANCHORS (Sentences above threshold):")
            for pos, score, sentence in thought_anchors:
                report.append(f"  Position {pos} (Score: {score:.3f})")
                report.append(f"     Text: {sentence[:100]}{'...' if len(sentence) > 100 else ''}")
                report.append("")
        
        # Answer consistency analysis
        consistency = self.analyze_answer_consistency(results)
        report.append("ANSWER CONSISTENCY ANALYSIS:")
        report.append(f"  High consistency sentences: {len(consistency['high_consistency'])}")
        report.append(f"  Low consistency sentences: {len(consistency['low_consistency'])}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report) 