"""
Sentence taxonomy for reasoning functions.
"""

from enum import Enum
from typing import Dict, List, Optional
import re

class SentenceCategory(Enum):
    """Enumeration of sentence categories in reasoning traces."""
    
    PROBLEM_SETUP = "problem_setup"
    PLAN_GENERATION = "plan_generation"
    FACT_RETRIEVAL = "fact_retrieval"
    ACTIVE_COMPUTATION = "active_computation"
    UNCERTAINTY_MANAGEMENT = "uncertainty_management"
    RESULT_CONSOLIDATION = "result_consolidation"
    SELF_CHECKING = "self_checking"
    FINAL_ANSWER_EMISSION = "final_answer_emission"

# Category descriptions from the paper
SENTENCE_CATEGORIES = {
    SentenceCategory.PROBLEM_SETUP: {
        "name": "Problem Setup",
        "description": "Parsing or rephrasing the problem",
        "examples": [
            "Let me understand the problem...",
            "The question asks us to...",
            "We need to find...",
        ]
    },
    SentenceCategory.PLAN_GENERATION: {
        "name": "Plan Generation", 
        "description": "Stating or deciding on a plan of action, meta-reasoning",
        "examples": [
            "Let me try a different approach...",
            "I'll solve this step by step...",
            "First, I need to...",
        ]
    },
    SentenceCategory.FACT_RETRIEVAL: {
        "name": "Fact Retrieval",
        "description": "Recalling facts, formulas, problem details without computation",
        "examples": [
            "I know that...",
            "The formula for this is...",
            "From the problem, we have...",
        ]
    },
    SentenceCategory.ACTIVE_COMPUTATION: {
        "name": "Active Computation",
        "description": "Algebra, calculations, or other manipulations toward the answer",
        "examples": [
            "2 + 3 = 5",
            "Let me calculate...",
            "Substituting the values...",
        ]
    },
    SentenceCategory.UNCERTAINTY_MANAGEMENT: {
        "name": "Uncertainty Management",
        "description": "Expressing confusion, re-evaluating, including backtracking",
        "examples": [
            "Wait, that doesn't seem right...",
            "I'm not sure about this...",
            "Let me reconsider...",
        ]
    },
    SentenceCategory.RESULT_CONSOLIDATION: {
        "name": "Result Consolidation",
        "description": "Aggregating intermediate results, summarizing, or preparing",
        "examples": [
            "So far we have...",
            "Putting this together...",
            "Now we can see that...",
        ]
    },
    SentenceCategory.SELF_CHECKING: {
        "name": "Self Checking",
        "description": "Verifying previous steps, checking calculations, and re-confirmations",
        "examples": [
            "Let me verify this...",
            "Double-checking my work...",
            "This looks correct because...",
        ]
    },
    SentenceCategory.FINAL_ANSWER_EMISSION: {
        "name": "Final Answer Emission",
        "description": "Explicitly stating the final answer",
        "examples": [
            "Therefore, the answer is...",
            "The final result is...",
            "So the solution is...",
        ]
    }
}

class SentenceTaxonomy:
    """Taxonomy for categorizing sentences in reasoning traces."""
    
    def __init__(self):
        self.categories = SENTENCE_CATEGORIES
        self._build_patterns()
    
    def _build_patterns(self):
        """Build regex patterns for automatic categorization."""
        self.patterns = {}
        
        # Problem setup patterns
        self.patterns[SentenceCategory.PROBLEM_SETUP] = [
            r"\b(understand|problem|question|asks|need to find|looking for)\b",
            r"\b(given|provided|stated)\b",
        ]
        
        # Plan generation patterns
        self.patterns[SentenceCategory.PLAN_GENERATION] = [
            r"\b(approach|strategy|plan|method|step by step|first|next|then)\b",
            r"\b(try|attempt|solve|figure out)\b",
        ]
        
        # Fact retrieval patterns
        self.patterns[SentenceCategory.FACT_RETRIEVAL] = [
            r"\b(know|remember|formula|rule|theorem|definition)\b",
            r"\b(from|given|provided|stated)\b",
        ]
        
        # Active computation patterns
        self.patterns[SentenceCategory.ACTIVE_COMPUTATION] = [
            r"\b(calculate|compute|solve|multiply|divide|add|subtract)\b",
            r"\b(=|plus|minus|times|divided by)\b",
            r"\b(substitute|plug in|evaluate)\b",
        ]
        
        # Uncertainty management patterns
        self.patterns[SentenceCategory.UNCERTAINTY_MANAGEMENT] = [
            r"\b(wait|hmm|not sure|confused|reconsider|backtrack)\b",
            r"\b(mistake|error|wrong|doesn't seem right)\b",
        ]
        
        # Result consolidation patterns
        self.patterns[SentenceCategory.RESULT_CONSOLIDATION] = [
            r"\b(so far|putting together|therefore|thus|hence)\b",
            r"\b(summarize|combine|aggregate)\b",
        ]
        
        # Self checking patterns
        self.patterns[SentenceCategory.SELF_CHECKING] = [
            r"\b(verify|check|double-check|confirm|validate)\b",
            r"\b(correct|right|accurate|precise)\b",
        ]
        
        # Final answer patterns
        self.patterns[SentenceCategory.FINAL_ANSWER_EMISSION] = [
            r"\b(answer|result|solution|final|therefore)\b",
            r"\b(the answer is|the result is|the solution is)\b",
        ]
    
    def categorize_sentence(self, sentence: str) -> SentenceCategory:
        """Categorize a sentence based on its content."""
        sentence_lower = sentence.lower().strip()
        
        # Score each category based on pattern matches
        scores = {}
        for category, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, sentence_lower):
                    score += 1
            scores[category] = score
        
        # Return the category with the highest score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            # Default to fact retrieval if no clear pattern
            return SentenceCategory.FACT_RETRIEVAL
    
    def get_category_description(self, category: SentenceCategory) -> str:
        """Get the description for a category."""
        return self.categories[category]["description"]
    
    def get_category_examples(self, category: SentenceCategory) -> List[str]:
        """Get example sentences for a category."""
        return self.categories[category]["examples"]
    
    def get_all_categories(self) -> List[SentenceCategory]:
        """Get all available categories."""
        return list(self.categories.keys())
    
    def validate_category(self, category: str) -> bool:
        """Validate if a category string is valid."""
        try:
            SentenceCategory(category)
            return True
        except ValueError:
            return False 