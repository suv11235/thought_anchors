"""
Sentence-level analysis for reasoning traces.
"""

from .taxonomy import SentenceTaxonomy, SENTENCE_CATEGORIES, SentenceCategory
from .labeling import SentenceLabeler, LabeledSentence
from typing import List, Dict, Any

class SentenceAnalyzer:
    """Main class for sentence-level analysis of reasoning traces."""
    
    def __init__(self, model=None, use_llm_labeling: bool = True):
        """
        Initialize the sentence analyzer.
        
        Args:
            model: LLM model for labeling (optional)
            use_llm_labeling: Whether to use LLM-based labeling or rule-based
        """
        self.taxonomy = SentenceTaxonomy()
        self.labeler = SentenceLabeler(model, use_llm_labeling)
    
    def analyze_reasoning_trace(self, text: str) -> List[LabeledSentence]:
        """Analyze a reasoning trace and return labeled sentences."""
        return self.labeler.label_reasoning_trace(text)
    
    def get_sentence_importance(self, labeled_sentences: List[LabeledSentence]) -> Dict[int, float]:
        """Get importance scores for sentences based on their categories."""
        importance_scores = {}
        
        # Define importance weights for different categories
        category_weights = {
            SentenceCategory.PLAN_GENERATION: 0.9,
            SentenceCategory.UNCERTAINTY_MANAGEMENT: 0.8,
            SentenceCategory.FINAL_ANSWER_EMISSION: 0.7,
            SentenceCategory.ACTIVE_COMPUTATION: 0.6,
            SentenceCategory.RESULT_CONSOLIDATION: 0.5,
            SentenceCategory.SELF_CHECKING: 0.4,
            SentenceCategory.FACT_RETRIEVAL: 0.3,
            SentenceCategory.PROBLEM_SETUP: 0.2,
        }
        
        for sentence in labeled_sentences:
            base_weight = category_weights.get(sentence.category, 0.5)
            confidence_multiplier = sentence.confidence
            importance_scores[sentence.position] = base_weight * confidence_multiplier
        
        return importance_scores
    
    def get_thought_anchors(self, labeled_sentences: List[LabeledSentence], 
                          threshold: float = 0.7) -> List[LabeledSentence]:
        """Identify thought anchors (high-importance sentences)."""
        importance_scores = self.get_sentence_importance(labeled_sentences)
        
        thought_anchors = []
        for sentence in labeled_sentences:
            if importance_scores.get(sentence.position, 0) >= threshold:
                thought_anchors.append(sentence)
        
        return thought_anchors
    
    def get_category_distribution(self, labeled_sentences: List[LabeledSentence]) -> Dict[str, int]:
        """Get the distribution of sentence categories."""
        distribution = {}
        for sentence in labeled_sentences:
            category = sentence.category.value
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def get_analysis_summary(self, labeled_sentences: List[LabeledSentence]) -> Dict[str, Any]:
        """Get a comprehensive summary of the analysis."""
        summary = {
            "total_sentences": len(labeled_sentences),
            "category_distribution": self.get_category_distribution(labeled_sentences),
            "importance_scores": self.get_sentence_importance(labeled_sentences),
            "thought_anchors": [s.position for s in self.get_thought_anchors(labeled_sentences)],
            "labeling_stats": self.labeler.get_labeling_stats(labeled_sentences),
        }
        
        return summary

__all__ = ["SentenceAnalyzer", "SentenceTaxonomy", "SentenceLabeler", "LabeledSentence", "SENTENCE_CATEGORIES"] 