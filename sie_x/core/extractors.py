"""
Extraction utilities and candidate generators for SIE-X.

This module provides helper classes for generating keyword candidates
and filtering noise from extracted terms.
"""

from typing import List, Set, Tuple
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class CandidateExtractor:
    """
    Generate keyword candidates from text using various extraction strategies.
    
    This class provides methods for extracting:
    - Named entities from spaCy docs
    - Meaningful noun phrases
    - Key phrases using regex patterns
    """
    
    def __init__(self):
        """Initialize the candidate extractor."""
        self.min_phrase_length = 2
        self.max_phrase_length = 50
    
    def extract_entities(self, doc) -> List[Tuple[str, str]]:
        """
        Extract named entities from spaCy doc.
        
        Args:
            doc: spaCy Doc object
        
        Returns:
            List of (text, label) tuples
            
        Example:
            >>> entities = extractor.extract_entities(doc)
            >>> # [("Apple Inc.", "ORG"), ("Tim Cook", "PERSON")]
        """
        entities = []
        
        for ent in doc.ents:
            # Filter out very short entities
            if len(ent.text.strip()) < 2:
                continue
                
            # Filter out entities that are just numbers
            if ent.text.strip().isdigit():
                continue
            
            entities.append((ent.text.strip(), ent.label_))
        
        logger.debug(f"Extracted {len(entities)} entities")
        return entities
    
    def extract_noun_phrases(self, doc) -> List[str]:
        """
        Extract meaningful noun phrases from spaCy doc.
        
        Filters out:
        - Pronouns (he, she, it)
        - Single determiners (the, a, an)
        - Very short or very long phrases
        
        Args:
            doc: spaCy Doc object
        
        Returns:
            List of noun phrase strings
        """
        noun_phrases = []
        
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            
            # Length filters
            if len(phrase) < self.min_phrase_length:
                continue
            if len(phrase) > self.max_phrase_length:
                continue
            
            # Filter out phrases that are all stopwords/punctuation
            if all(token.is_stop or token.is_punct for token in chunk):
                continue
            
            # Filter out lone pronouns
            if len(chunk) == 1 and chunk[0].pos_ == "PRON":
                continue
            
            # Filter out lone determiners
            if len(chunk) == 1 and chunk[0].pos_ == "DET":
                continue
            
            noun_phrases.append(phrase)
        
        logger.debug(f"Extracted {len(noun_phrases)} noun phrases")
        return noun_phrases
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases using regex patterns.
        
        Patterns matched:
        - "X of Y" (e.g., "President of France")
        - "X and Y" (e.g., "machine learning and AI")
        - Quoted phrases (e.g., "artificial intelligence")
        
        Args:
            text: Input text string
        
        Returns:
            List of extracted phrases
        """
        phrases = []
        
        # Pattern: "X of Y"
        of_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        of_matches = re.findall(of_pattern, text)
        phrases.extend([f"{x} of {y}" for x, y in of_matches])
        
        # Pattern: "X and Y" (capitalized words)
        and_pattern = r'\b([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\b'
        and_matches = re.findall(and_pattern, text)
        phrases.extend([f"{x} and {y}" for x, y in and_matches])
        
        # Pattern: Quoted phrases
        quote_pattern = r'"([^"]+)"'
        quote_matches = re.findall(quote_pattern, text)
        phrases.extend(quote_matches)
        
        # Pattern: Phrases in parentheses (often definitions/clarifications)
        paren_pattern = r'\(([^)]+)\)'
        paren_matches = re.findall(paren_pattern, text)
        # Only include if it looks like a proper phrase (not just numbers or single words)
        for match in paren_matches:
            if len(match.split()) >= 2 and not match.isdigit():
                phrases.append(match)
        
        # Filter duplicates while preserving order
        seen = set()
        unique_phrases = []
        for phrase in phrases:
            phrase_clean = phrase.strip()
            if phrase_clean and phrase_clean not in seen:
                seen.add(phrase_clean)
                unique_phrases.append(phrase_clean)
        
        logger.debug(f"Extracted {len(unique_phrases)} key phrases")
        return unique_phrases
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent matching.
        
        - Lowercase
        - Remove extra whitespace
        - Keep meaningful punctuation (. , -)
        
        Args:
            text: Input text
        
        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text


class TermFilter:
    """
    Filter out noise and invalid terms from candidate keywords.
    
    Applies various heuristics to determine if a candidate term
    is likely to be a meaningful keyword.
    """
    
    def __init__(
        self,
        min_length: int = 2,
        max_length: int = 50,
        custom_stop_patterns: List[str] = None
    ):
        """
        Initialize the term filter.
        
        Args:
            min_length: Minimum character length for valid terms
            max_length: Maximum character length for valid terms
            custom_stop_patterns: Additional regex patterns to filter out
        """
        self.min_length = min_length
        self.max_length = max_length
        
        # Default stop patterns
        self.stop_patterns = [
            r'^\d+$',              # Only numbers
            r'^[^\w]+$',           # Only punctuation
            r'^[a-z]$',            # Single lowercase letter
            r'^\s+$',              # Only whitespace
            r'^(http|www)',        # URLs
            r'^[@#]\w+',           # Social media handles/hashtags
        ]
        
        # Add custom patterns if provided
        if custom_stop_patterns:
            self.stop_patterns.extend(custom_stop_patterns)
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.stop_patterns]
        
        # Common noise words/phrases to filter
        self.noise_terms = {
            'click here', 'read more', 'learn more', 'sign up', 'log in',
            'next page', 'previous page', 'home page', 'contact us',
            'terms of service', 'privacy policy', 'copyright', 'all rights reserved'
        }
    
    def is_valid(self, term: str) -> bool:
        """
        Check if a term is valid (not noise).
        
        Args:
            term: Term to validate
        
        Returns:
            True if valid, False if should be filtered out
        """
        # Length check
        if len(term) < self.min_length or len(term) > self.max_length:
            return False
        
        # Check against stop patterns
        for pattern in self.compiled_patterns:
            if pattern.match(term):
                return False
        
        # Check against noise terms
        if term.lower() in self.noise_terms:
            return False
        
        # Check if it's mostly punctuation (>50%)
        if term:
            punct_ratio = sum(1 for c in term if not c.isalnum() and not c.isspace()) / len(term)
            if punct_ratio > 0.5:
                return False
        
        return True
    
    def filter_candidates(self, candidates: List[str]) -> List[str]:
        """
        Apply all filters to a list of candidate terms.
        
        Args:
            candidates: List of candidate terms
        
        Returns:
            Filtered list of valid terms
        """
        valid_candidates = [c for c in candidates if self.is_valid(c)]
        
        logger.debug(
            f"Filtered {len(candidates)} candidates to {len(valid_candidates)} valid terms "
            f"({len(candidates) - len(valid_candidates)} removed)"
        )
        
        return valid_candidates
    
    def filter_by_frequency(
        self,
        candidates: List[str],
        min_frequency: int = 1,
        max_frequency: int = None
    ) -> List[str]:
        """
        Filter candidates by frequency of occurrence.
        
        Args:
            candidates: List of candidate terms (can have duplicates)
            min_frequency: Minimum number of occurrences
            max_frequency: Maximum number of occurrences (optional)
        
        Returns:
            List of candidates that meet frequency criteria
        """
        # Count frequencies
        freq = Counter(candidates)
        
        # Filter by frequency
        filtered = []
        for term, count in freq.items():
            if count >= min_frequency:
                if max_frequency is None or count <= max_frequency:
                    filtered.append(term)
        
        return filtered


def merge_overlapping_phrases(phrases: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Merge overlapping phrases, keeping the one with higher score.
    
    For example, if we have both "machine learning" and "learning",
    we keep only "machine learning" (assuming it has higher score).
    
    Args:
        phrases: List of (phrase, score) tuples
    
    Returns:
        Filtered list with overlaps removed
    """
    if not phrases:
        return []
    
    # Sort by score (descending) so we keep higher-scored phrases
    sorted_phrases = sorted(phrases, key=lambda x: x[1], reverse=True)
    
    # Track which phrases to keep
    keep = []
    seen_words = set()
    
    for phrase, score in sorted_phrases:
        words = set(phrase.lower().split())
        
        # Check if this phrase's words overlap significantly with seen phrases
        overlap = len(words & seen_words) / len(words) if words else 0
        
        # Keep if less than 80% overlap with existing phrases
        if overlap < 0.8:
            keep.append((phrase, score))
            seen_words.update(words)
    
    # Sort back by score
    return sorted(keep, key=lambda x: x[1], reverse=True)


def deduplicate_phrases(phrases: List[str]) -> List[str]:
    """
    Remove duplicate phrases (case-insensitive).
    
    Args:
        phrases: List of phrases
    
    Returns:
        Deduplicated list, preserving first occurrence
    """
    seen = set()
    unique = []
    
    for phrase in phrases:
        phrase_lower = phrase.lower()
        if phrase_lower not in seen:
            seen.add(phrase_lower)
            unique.append(phrase)
    
    return unique
