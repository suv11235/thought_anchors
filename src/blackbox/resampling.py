"""
Black-box resampling for counterfactual importance analysis.
"""

import json
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

@dataclass
class ResamplingResult:
    """Result of a resampling experiment."""
    original_sentence: str
    original_answer: str
    resampled_answers: List[str]
    answer_distribution: Dict[str, int]
    importance_score: float
    metadata: Dict[str, Any]

class BlackboxResampler:
    """Black-box resampling for measuring counterfactual importance."""
    
    def __init__(self, model, num_rollouts: int = 100, 
                 temperature: float = 0.8, top_p: float = 0.9):
        """
        Initialize the black-box resampler.
        
        Args:
            model: The language model to use for resampling
            num_rollouts: Number of rollouts per sentence
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
        """
        self.model = model
        self.num_rollouts = num_rollouts
        self.temperature = temperature
        self.top_p = top_p
    
    def resample_sentence(self, reasoning_trace: str, sentence_position: int,
                         sentence_text: str) -> ResamplingResult:
        """
        Resample a specific sentence and measure its impact on the final answer.
        
        Args:
            reasoning_trace: The full reasoning trace
            sentence_position: Position of the sentence to resample
            sentence_text: Text of the sentence to resample
            
        Returns:
            ResamplingResult with importance analysis
        """
        # Split the reasoning trace into sentences
        sentences = self._split_into_sentences(reasoning_trace)
        
        if sentence_position >= len(sentences):
            raise ValueError(f"Sentence position {sentence_position} out of range")
        
        # Get the context up to the sentence position
        context = self._get_context_up_to_position(sentences, sentence_position)
        
        # Generate alternative sentences
        alternative_sentences = self._generate_alternative_sentences(
            context, sentence_text, self.num_rollouts
        )
        
        # Generate answers for each alternative
        resampled_answers = []
        for alt_sentence in tqdm(alternative_sentences, desc=f"Resampling sentence {sentence_position}"):
            # Create the modified reasoning trace
            modified_trace = self._create_modified_trace(
                sentences, sentence_position, alt_sentence
            )
            
            # Generate answer from the modified trace
            answer = self._generate_answer(modified_trace)
            resampled_answers.append(answer)
        
        # Get the original answer
        original_answer = self._generate_answer(reasoning_trace)
        
        # Calculate answer distribution
        answer_distribution = self._calculate_answer_distribution(resampled_answers)
        
        # Calculate importance score
        importance_score = self._calculate_importance_score(
            original_answer, resampled_answers
        )
        
        return ResamplingResult(
            original_sentence=sentence_text,
            original_answer=original_answer,
            resampled_answers=resampled_answers,
            answer_distribution=answer_distribution,
            importance_score=importance_score,
            metadata={
                "sentence_position": sentence_position,
                "num_rollouts": self.num_rollouts,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
        )
    
    def resample_all_sentences(self, reasoning_trace: str) -> List[ResamplingResult]:
        """
        Resample all sentences in a reasoning trace.
        
        Args:
            reasoning_trace: The full reasoning trace
            
        Returns:
            List of ResamplingResult for each sentence
        """
        sentences = self._split_into_sentences(reasoning_trace)
        results = []
        
        for i, sentence in enumerate(sentences):
            try:
                result = self.resample_sentence(reasoning_trace, i, sentence)
                results.append(result)
            except Exception as e:
                print(f"Error resampling sentence {i}: {e}")
                continue
        
        return results
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_context_up_to_position(self, sentences: List[str], position: int) -> str:
        """Get context up to a specific sentence position."""
        return " ".join(sentences[:position])
    
    def _generate_alternative_sentences(self, context: str, original_sentence: str,
                                      num_alternatives: int) -> List[str]:
        """Generate alternative sentences using the model."""
        prompt = f"""Given this context:
{context}

The next sentence was: "{original_sentence}"

Generate {num_alternatives} different sentences that could naturally follow this context. 
Each sentence should be a single, complete sentence that makes sense in the reasoning flow.
Make sure the sentences are diverse and different from the original.

Return only the sentences, one per line:"""

        try:
            response = self._call_model(prompt)
            alternatives = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Ensure we have enough alternatives
            while len(alternatives) < num_alternatives:
                # Generate more alternatives
                additional_response = self._call_model(prompt)
                additional = [line.strip() for line in additional_response.split('\n') if line.strip()]
                alternatives.extend(additional)
            
            return alternatives[:num_alternatives]
            
        except Exception as e:
            print(f"Error generating alternatives: {e}")
            # Fallback: create simple variations
            return self._create_simple_alternatives(original_sentence, num_alternatives)
    
    def _create_simple_alternatives(self, original_sentence: str, num_alternatives: int) -> List[str]:
        """Create simple alternatives when model generation fails."""
        alternatives = []
        
        # Simple variations
        variations = [
            f"Let me think about this differently.",
            f"I need to approach this step by step.",
            f"Let me try a different method.",
            f"I should reconsider this approach.",
            f"Let me break this down further.",
            f"I need to be more careful here.",
            f"Let me verify this step.",
            f"I should double-check this.",
        ]
        
        for i in range(num_alternatives):
            if i < len(variations):
                alternatives.append(variations[i])
            else:
                # Create a generic alternative
                alternatives.append(f"Let me continue with step {i+1}.")
        
        return alternatives
    
    def _create_modified_trace(self, sentences: List[str], position: int, 
                             new_sentence: str) -> str:
        """Create a modified reasoning trace with a new sentence."""
        modified_sentences = sentences.copy()
        modified_sentences[position] = new_sentence
        return " ".join(modified_sentences)
    
    def _generate_answer(self, reasoning_trace: str) -> str:
        """Generate an answer from a reasoning trace."""
        prompt = f"""Based on this reasoning trace, what is the final answer?

{reasoning_trace}

Final answer:"""

        try:
            response = self._call_model(prompt)
            # Extract just the answer part
            answer = response.strip()
            return answer
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "ERROR: Could not generate answer"
    
    def _call_model(self, prompt: str) -> str:
        """Call the language model with a prompt."""
        # This is a placeholder - implementation depends on the model type
        if hasattr(self.model, 'generate'):
            # Local model
            response = self.model.generate(
                prompt, 
                max_tokens=200,
                temperature=self.temperature,
                top_p=self.top_p
            )
            return response
        else:
            # API model
            raise NotImplementedError("Model calling not implemented for this model type")
    
    def _calculate_answer_distribution(self, answers: List[str]) -> Dict[str, int]:
        """Calculate the distribution of answers."""
        distribution = {}
        for answer in answers:
            # Normalize answer for comparison
            normalized = self._normalize_answer(answer)
            distribution[normalized] = distribution.get(normalized, 0) + 1
        return distribution
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        # Remove common prefixes and normalize
        answer = answer.strip().lower()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "the answer is", "answer:", "final answer:", "result:", "solution:"
        ]
        
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        return answer
    
    def _calculate_importance_score(self, original_answer: str, 
                                  resampled_answers: List[str]) -> float:
        """Calculate importance score based on answer consistency."""
        normalized_original = self._normalize_answer(original_answer)
        
        # Count how many resampled answers match the original
        matching_count = 0
        for resampled_answer in resampled_answers:
            normalized_resampled = self._normalize_answer(resampled_answer)
            if normalized_resampled == normalized_original:
                matching_count += 1
        
        # Importance score is the fraction of answers that changed
        importance_score = 1.0 - (matching_count / len(resampled_answers))
        
        return importance_score
    
    def save_results(self, results: List[ResamplingResult], filepath: str):
        """Save resampling results to a JSON file."""
        data = []
        for result in results:
            data.append({
                "original_sentence": result.original_sentence,
                "original_answer": result.original_answer,
                "resampled_answers": result.resampled_answers,
                "answer_distribution": result.answer_distribution,
                "importance_score": result.importance_score,
                "metadata": result.metadata
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_results(self, filepath: str) -> List[ResamplingResult]:
        """Load resampling results from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results = []
        for item in data:
            result = ResamplingResult(
                original_sentence=item["original_sentence"],
                original_answer=item["original_answer"],
                resampled_answers=item["resampled_answers"],
                answer_distribution=item["answer_distribution"],
                importance_score=item["importance_score"],
                metadata=item["metadata"]
            )
            results.append(result)
        
        return results 