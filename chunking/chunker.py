"""
Advanced document chunking with semantic boundaries.
"""

from typing import List, Optional, Tuple
import numpy as np
from transformers import PreTrainedTokenizer


class DocumentChunker:
    """Intelligent document chunking with overlap and semantic boundaries."""

    def __init__(
            self,
            max_tokens: int = 512,
            overlap_ratio: float = 0.1,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            respect_sentences: bool = True
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = int(max_tokens * overlap_ratio)
        self.tokenizer = tokenizer
        self.respect_sentences = respect_sentences

    def chunk(self, text: str) -> List[str]:
        """Chunk document into overlapping segments."""
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.max_tokens:
            return [text]

        # Find sentence boundaries if requested
        if self.respect_sentences:
            boundaries = self._find_sentence_boundaries(text, tokens)
        else:
            boundaries = []

        # Create chunks
        chunks = []
        start = 0

        while start < len(tokens):
            # Find end position
            end = start + self.max_tokens

            # Adjust for sentence boundary
            if boundaries and end < len(tokens):
                # Find nearest sentence boundary
                nearest = min(boundaries,
                              key=lambda x: abs(x - end) if x <= end else float('inf'))
                if nearest > start:
                    end = nearest

            # Extract chunk
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

            # Move start with overlap
            start = end - self.overlap_tokens

            # Ensure progress
            if start >= end:
                start = end

        return chunks

    def _find_sentence_boundaries(self, text: str, tokens: List[int]) -> List[int]:
        """Find token positions that correspond to sentence boundaries."""
        # Simple implementation - would use spaCy in production
        boundaries = []

        decoded_tokens = [self.tokenizer.decode([t]) for t in tokens]

        for i, token_text in enumerate(decoded_tokens):
            if any(punct in token_text for punct in ['.', '!', '?']):
                boundaries.append(i + 1)

        return boundaries


class SlidingWindowChunker(DocumentChunker):
    """Enhanced chunker with dynamic window sizing."""

    def __init__(self, *args, adaptive_size: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_size = adaptive_size

    def chunk(self, text: str) -> List[str]:
        """Chunk with adaptive window sizing based on content density."""
        if not self.adaptive_size:
            return super().chunk(text)

        # Analyze content density
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        densities = self._calculate_token_densities(tokens)

        # Create variable-sized chunks based on density
        chunks = []
        start = 0

        while start < len(tokens):
            # Determine chunk size based on local density
            local_density = np.mean(densities[start:start + 50])
            chunk_size = int(self.max_tokens * (0.7 + 0.3 * local_density))

            end = min(start + chunk_size, len(tokens))

            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

            start = end - self.overlap_tokens

        return chunks

    def _calculate_token_densities(self, tokens: List[int]) -> np.ndarray:
        """Calculate information density for each token."""
        # Simplified: use token frequency as proxy for density
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        # Inverse frequency as density (rare = dense)
        max_count = max(token_counts.values())
        densities = [
            1.0 - (token_counts[token] / max_count)
            for token in tokens
        ]

        return np.array(densities)