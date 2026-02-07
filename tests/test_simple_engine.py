"""Tests for SimpleSemanticEngine with mocked ML models."""

import pytest
from unittest.mock import patch, MagicMock

from sie_x.core.models import Keyword
from .conftest import MockSentenceTransformer, MockSpacy


def _make_engine():
    """Create a SimpleSemanticEngine with mocked models."""
    from sie_x.core.simple_engine import SimpleSemanticEngine
    return SimpleSemanticEngine()


class TestEngineInit:

    @patch('sie_x.core.simple_engine.SentenceTransformer', MockSentenceTransformer)
    @patch('sie_x.core.simple_engine.load_spacy_model', MockSpacy)
    def test_initializes(self):
        engine = _make_engine()
        assert engine is not None
        assert engine.model_name == "sentence-transformers/all-MiniLM-L6-v2"


class TestExtract:

    @patch('sie_x.core.simple_engine.SentenceTransformer', MockSentenceTransformer)
    @patch('sie_x.core.simple_engine.load_spacy_model', MockSpacy)
    def test_returns_keyword_list(self):
        engine = _make_engine()
        result = engine.extract("Machine learning and Python programming")
        assert isinstance(result, list)
        assert all(isinstance(kw, Keyword) for kw in result)

    @patch('sie_x.core.simple_engine.SentenceTransformer', MockSentenceTransformer)
    @patch('sie_x.core.simple_engine.load_spacy_model', MockSpacy)
    def test_empty_text_returns_empty(self):
        engine = _make_engine()
        assert engine.extract("") == []

    @patch('sie_x.core.simple_engine.SentenceTransformer', MockSentenceTransformer)
    @patch('sie_x.core.simple_engine.load_spacy_model', MockSpacy)
    def test_whitespace_returns_empty(self):
        engine = _make_engine()
        assert engine.extract("   ") == []

    @patch('sie_x.core.simple_engine.SentenceTransformer', MockSentenceTransformer)
    @patch('sie_x.core.simple_engine.load_spacy_model', MockSpacy)
    def test_top_k_limits_results(self):
        engine = _make_engine()
        result = engine.extract("Machine learning Python data", top_k=1)
        assert len(result) <= 1

    @patch('sie_x.core.simple_engine.SentenceTransformer', MockSentenceTransformer)
    @patch('sie_x.core.simple_engine.load_spacy_model', MockSpacy)
    def test_min_confidence_filters(self):
        engine = _make_engine()
        result = engine.extract(
            "Machine learning Python data",
            min_confidence=0.99,
        )
        assert all(kw.confidence >= 0.99 for kw in result)

    @patch('sie_x.core.simple_engine.SentenceTransformer', MockSentenceTransformer)
    @patch('sie_x.core.simple_engine.load_spacy_model', MockSpacy)
    def test_results_sorted_by_score(self):
        engine = _make_engine()
        result = engine.extract("Machine learning Python data")
        if len(result) > 1:
            scores = [kw.score for kw in result]
            assert scores == sorted(scores, reverse=True)


class TestCacheAndStats:

    @patch('sie_x.core.simple_engine.SentenceTransformer', MockSentenceTransformer)
    @patch('sie_x.core.simple_engine.load_spacy_model', MockSpacy)
    def test_get_stats(self):
        engine = _make_engine()
        stats = engine.get_stats()
        assert "model_name" in stats
        assert "spacy_model" in stats
        assert "cache_size" in stats

    @patch('sie_x.core.simple_engine.SentenceTransformer', MockSentenceTransformer)
    @patch('sie_x.core.simple_engine.load_spacy_model', MockSpacy)
    def test_clear_cache(self):
        engine = _make_engine()
        engine.extract("Machine learning test")
        engine.clear_cache()
        assert engine.get_stats()["cache_size"] == 0
