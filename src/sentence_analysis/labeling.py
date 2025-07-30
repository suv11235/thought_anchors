"""
LLM-based sentence labeling for reasoning traces.
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .taxonomy import SentenceCategory, SENTENCE_CATEGORIES

@dataclass
class LabeledSentence:
    """A sentence with its category label."""
    text: str
    category: SentenceCategory
    confidence: float
    position: int
    metadata: Optional[Dict[str, Any]] = None

class SentenceLabeler:
    """LLM-based sentence labeler for reasoning traces."""
    
    def __init__(self, model=None, use_llm_labeling: bool = True):
        """
        Initialize the sentence labeler.
        
        Args:
            model: LLM model for labeling (optional)
            use_llm_labeling: Whether to use LLM-based labeling or rule-based
        """
        self.model = model
        self.use_llm_labeling = use_llm_labeling
        self._build_labeling_prompt()
    
    def _build_labeling_prompt(self):
        """Build the prompt for LLM-based labeling."""
        categories_text = "\n".join([
            f"{i+1}. {cat.value}: {SENTENCE_CATEGORIES[cat]['description']}"
            for i, cat in enumerate(SENTENCE_CATEGORIES.keys())
        ])
        
        self.labeling_prompt = f"""You are an expert at categorizing sentences in reasoning traces. 
Given a sentence from a reasoning trace, categorize it into one of the following categories:

{categories_text}

For each sentence, respond with only the category name (e.g., "plan_generation") and a confidence score from 0.0 to 1.0.

Example:
Sentence: "Let me try a different approach to solve this problem."
Response: {{"category": "plan_generation", "confidence": 0.9}}

Sentence: "2 + 3 = 5"
Response: {{"category": "active_computation", "confidence": 0.95}}

Now categorize this sentence:
Sentence: "{{sentence}}"
Response:"""
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLP-aware splitting."""
        # Simple sentence splitting - can be enhanced with NLTK or spaCy
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def label_sentence_llm(self, sentence: str) -> Dict[str, Any]:
        """Label a sentence using LLM."""
        if not self.model:
            raise ValueError("No model provided for LLM labeling")
        
        prompt = self.labeling_prompt.format(sentence=sentence)
        
        try:
            # This would be implemented based on the specific model being used
            response = self._call_llm(prompt)
            result = json.loads(response)
            
            # Validate the response
            if "category" not in result or "confidence" not in result:
                raise ValueError("Invalid response format")
            
            category = result["category"]
            confidence = float(result["confidence"])
            
            # Validate category
            if not self._validate_category(category):
                raise ValueError(f"Invalid category: {category}")
            
            return {
                "category": SentenceCategory(category),
                "confidence": confidence
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback to rule-based labeling
            return self._label_sentence_rule_based(sentence)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        # This is a placeholder - would be implemented based on the model type
        if hasattr(self.model, 'generate'):
            # Local model
            response = self.model.generate(prompt, max_tokens=100)
            return response
        else:
            # API model (OpenAI, Anthropic, etc.)
            # Implementation would depend on the specific API
            raise NotImplementedError("LLM calling not implemented for this model type")
    
    def _label_sentence_rule_based(self, sentence: str) -> Dict[str, Any]:
        """Label a sentence using rule-based approach."""
        from .taxonomy import SentenceTaxonomy
        
        taxonomy = SentenceTaxonomy()
        category = taxonomy.categorize_sentence(sentence)
        
        # Simple confidence based on pattern matches
        confidence = 0.7  # Default confidence for rule-based
        
        return {
            "category": category,
            "confidence": confidence
        }
    
    def _validate_category(self, category: str) -> bool:
        """Validate if a category string is valid."""
        try:
            SentenceCategory(category)
            return True
        except ValueError:
            return False
    
    def label_reasoning_trace(self, text: str) -> List[LabeledSentence]:
        """Label all sentences in a reasoning trace."""
        sentences = self.split_into_sentences(text)
        labeled_sentences = []
        
        for i, sentence in enumerate(sentences):
            if self.use_llm_labeling and self.model:
                result = self.label_sentence_llm(sentence)
            else:
                result = self._label_sentence_rule_based(sentence)
            
            labeled_sentence = LabeledSentence(
                text=sentence,
                category=result["category"],
                confidence=result["confidence"],
                position=i,
                metadata={"method": "llm" if self.use_llm_labeling else "rule_based"}
            )
            
            labeled_sentences.append(labeled_sentence)
        
        return labeled_sentences
    
    def get_labeling_stats(self, labeled_sentences: List[LabeledSentence]) -> Dict[str, Any]:
        """Get statistics about the labeling results."""
        stats = {
            "total_sentences": len(labeled_sentences),
            "category_counts": {},
            "average_confidence": 0.0,
            "confidence_distribution": {}
        }
        
        if not labeled_sentences:
            return stats
        
        # Count categories
        for sentence in labeled_sentences:
            category = sentence.category.value
            stats["category_counts"][category] = stats["category_counts"].get(category, 0) + 1
        
        # Calculate average confidence
        total_confidence = sum(s.confidence for s in labeled_sentences)
        stats["average_confidence"] = total_confidence / len(labeled_sentences)
        
        # Confidence distribution
        confidence_ranges = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        for low, high in confidence_ranges:
            count = sum(1 for s in labeled_sentences if low <= s.confidence < high)
            stats["confidence_distribution"][f"{low}-{high}"] = count
        
        return stats
    
    def save_labeled_sentences(self, labeled_sentences: List[LabeledSentence], filepath: str):
        """Save labeled sentences to a JSON file."""
        data = []
        for sentence in labeled_sentences:
            data.append({
                "text": sentence.text,
                "category": sentence.category.value,
                "confidence": sentence.confidence,
                "position": sentence.position,
                "metadata": sentence.metadata or {}
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_labeled_sentences(self, filepath: str) -> List[LabeledSentence]:
        """Load labeled sentences from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        labeled_sentences = []
        for item in data:
            labeled_sentence = LabeledSentence(
                text=item["text"],
                category=SentenceCategory(item["category"]),
                confidence=item["confidence"],
                position=item["position"],
                metadata=item.get("metadata", {})
            )
            labeled_sentences.append(labeled_sentence)
        
        return labeled_sentences 