"""
Shared fixtures and mocks for SIE-X test suite.

All mocks are designed to run without GPU, Redis, spaCy models
or SentenceTransformer downloads.
"""

import pytest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from sie_x.core.models import (
    Keyword,
    ExtractionOptions,
    ExtractionRequest,
    ExtractionResponse,
)


# ============================================================================
# MOCK IMPLEMENTATIONS
# ============================================================================

class MockSpacyToken:
    """Mock spaCy Token."""
    def __init__(self, text, is_stop=False, is_punct=False, pos_="NOUN"):
        self.text = text
        self.is_stop = is_stop
        self.is_punct = is_punct
        self.pos_ = pos_


class MockSpacyDoc:
    """Mock spaCy Doc object."""

    def __init__(self, text: str):
        self.text = text
        self.ents = []
        self.noun_chunks = []

    def add_entity(self, text: str, label: str, start: int, end: int):
        ent = Mock()
        ent.text = text
        ent.label_ = label
        ent.start_char = start
        ent.end_char = end
        self.ents.append(ent)

    def add_noun_chunk(self, text: str, start: int, end: int):
        chunk = Mock()
        chunk.text = text
        chunk.start_char = start
        chunk.end_char = end
        tokens = [MockSpacyToken(w) for w in text.split()]
        chunk.__iter__ = lambda s: iter(tokens)
        chunk.__len__ = lambda s: len(tokens)
        chunk.__getitem__ = lambda s, i: tokens[i]
        self.noun_chunks.append(chunk)


class MockSpacy:
    """Mock spaCy NLP pipeline."""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name

    def __call__(self, text: str) -> MockSpacyDoc:
        doc = MockSpacyDoc(text)
        if "machine learning" in text.lower():
            doc.add_entity("machine learning", "CONCEPT", 0, 16)
        if "python" in text.lower():
            doc.add_entity("Python", "PRODUCT", 0, 6)
        words = text.split()
        for i, word in enumerate(words[:3]):
            if len(word) > 3:
                doc.add_noun_chunk(word, i * 10, i * 10 + len(word))
        return doc


class MockSentenceTransformer:
    """Mock Sentence Transformer — returns deterministic embeddings."""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name

    def encode(
        self,
        sentences: List[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        rng = np.random.default_rng(42)
        n = len(sentences) if isinstance(sentences, list) else 1
        embeddings = rng.standard_normal((n, 384))
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings


class MockEngine:
    """Mock extraction engine for API/SDK tests."""

    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "mock-model")
        self.spacy_model = kwargs.get("spacy_model", "mock-spacy")
        self.call_count = 0

    def extract(
        self,
        text: str,
        top_k: int = 10,
        min_confidence: float = 0.3,
        include_entities: bool = True,
        include_concepts: bool = True,
    ) -> List[Keyword]:
        self.call_count += 1
        if not text or not text.strip():
            return []
        keywords = [
            Keyword(text="machine learning", score=0.92, type="CONCEPT",
                    count=2, confidence=0.88,
                    metadata={"source": "entity"}),
            Keyword(text="artificial intelligence", score=0.85, type="CONCEPT",
                    count=1, confidence=0.80,
                    metadata={"source": "noun_phrase"}),
        ]
        filtered = [kw for kw in keywords if kw.confidence >= min_confidence]
        return filtered[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "spacy_model": self.spacy_model,
            "cache_size": 0,
            "call_count": self.call_count,
        }

    def clear_cache(self):
        pass


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_text():
    return (
        "Machine learning is a subset of artificial intelligence that enables "
        "computers to learn from data. Python is a popular programming language "
        "for machine learning applications."
    )


@pytest.fixture
def sample_keywords():
    return [
        Keyword(text="machine learning", score=0.92, type="CONCEPT"),
        Keyword(text="artificial intelligence", score=0.88, type="CONCEPT"),
        Keyword(text="Python", score=0.85, type="PRODUCT"),
        Keyword(text="data", score=0.75, type="CONCEPT"),
    ]


@pytest.fixture
def mock_engine():
    return MockEngine()
