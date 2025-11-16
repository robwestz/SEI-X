"""
Standalone tests for SIE-X core components.

These tests can run without heavy ML models using mocks.
For integration tests with real models, see tests/integration/.
"""

import pytest
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import numpy as np

# Import core models
from sie_x.core.models import (
    Keyword,
    ExtractionOptions,
    ExtractionRequest,
    ExtractionResponse,
    BatchExtractionRequest,
    HealthResponse,
)


# ============================================================================
# MOCK IMPLEMENTATIONS
# ============================================================================

class MockSpacyDoc:
    """Mock spaCy Doc object for testing."""
    
    def __init__(self, text: str):
        self.text = text
        self.ents = []
        self.noun_chunks = []
    
    def add_entity(self, text: str, label: str, start: int, end: int):
        """Add a mock entity."""
        ent = Mock()
        ent.text = text
        ent.label_ = label
        ent.start_char = start
        ent.end_char = end
        self.ents.append(ent)
    
    def add_noun_chunk(self, text: str, start: int, end: int):
        """Add a mock noun chunk."""
        chunk = Mock()
        chunk.text = text
        chunk.start_char = start
        chunk.end_char = end
        # Mock tokens
        chunk.__iter__ = lambda self: iter([])
        self.noun_chunks.append(chunk)


class MockSpacy:
    """Mock spaCy NLP pipeline."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def __call__(self, text: str) -> MockSpacyDoc:
        """Process text and return mock doc."""
        doc = MockSpacyDoc(text)
        
        # Add some mock entities
        if "machine learning" in text.lower():
            doc.add_entity("machine learning", "CONCEPT", 0, 16)
        if "python" in text.lower():
            doc.add_entity("Python", "PRODUCT", 0, 6)
        
        # Add some mock noun chunks
        words = text.split()
        for i, word in enumerate(words[:3]):
            if len(word) > 3:
                doc.add_noun_chunk(word, i * 10, i * 10 + len(word))
        
        return doc


class MockSentenceTransformer:
    """Mock Sentence Transformer for testing."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def encode(
        self,
        sentences: List[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """Return mock embeddings."""
        # Create random embeddings (deterministic for testing)
        rng = np.random.default_rng(42)
        n = len(sentences) if isinstance(sentences, list) else 1
        embeddings = rng.standard_normal((n, 384))  # 384-dim like all-MiniLM

        # Normalize to unit vectors
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings


class MockEngine:
    """
    Mock extraction engine for testing without ML dependencies.
    
    Use this in API/SDK/CLI tests to avoid loading heavy models.
    """
    
    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "mock")
        self.spacy_model = kwargs.get("spacy_model", "mock")
        self.call_count = 0
    
    def extract(
        self,
        text: str,
        top_k: int = 10,
        min_confidence: float = 0.3,
        include_entities: bool = True,
        include_concepts: bool = True
    ) -> List[Keyword]:
        """Return mock keywords for testing."""
        self.call_count += 1
        
        # Generate predictable mock keywords
        keywords = []
        
        if include_entities:
            keywords.append(Keyword(
                text="Machine Learning",
                score=0.92,
                type="CONCEPT",
                count=2,
                confidence=0.88,
                positions=[(0, 16)],
                metadata={"source": "entity"}
            ))
        
        if include_concepts:
            keywords.append(Keyword(
                text="artificial intelligence",
                score=0.85,
                type="CONCEPT",
                count=1,
                confidence=0.80,
                positions=[(20, 43)],
                metadata={"source": "noun_phrase"}
            ))
            
            keywords.append(Keyword(
                text="neural networks",
                score=0.78,
                type="CONCEPT",
                count=1,
                confidence=0.75,
                metadata={"source": "noun_phrase"}
            ))
        
        # Filter by confidence and return top_k
        filtered = [kw for kw in keywords if kw.confidence >= min_confidence]
        return filtered[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Return mock stats."""
        return {
            "model_name": self.model_name,
            "spacy_model": self.spacy_model,
            "cache_size": 0,
            "call_count": self.call_count
        }
    
    def clear_cache(self):
        """Mock cache clearing."""
        pass


# ============================================================================
# MODEL TESTS
# ============================================================================

class TestKeyword:
    """Test Keyword model validation and serialization."""
    
    def test_keyword_creation(self):
        """Test creating a valid keyword."""
        kw = Keyword(
            text="machine learning",
            score=0.85,
            type="CONCEPT"
        )
        assert kw.text == "machine learning"
        assert kw.score == 0.85
        assert kw.type == "CONCEPT"
        assert kw.count == 1
        assert kw.confidence == 1.0
    
    def test_keyword_score_validation(self):
        """Test score must be 0-1."""
        with pytest.raises(ValueError):
            Keyword(text="test", score=1.5, type="CONCEPT")
        
        with pytest.raises(ValueError):
            Keyword(text="test", score=-0.1, type="CONCEPT")
    
    def test_keyword_confidence_validation(self):
        """Test confidence must be 0-1."""
        with pytest.raises(ValueError):
            Keyword(text="test", score=0.5, type="CONCEPT", confidence=2.0)
    
    def test_keyword_str_representation(self):
        """Test string representation."""
        kw = Keyword(text="test", score=0.75, type="CONCEPT")
        assert "test" in str(kw)
        assert "0.75" in str(kw)
    
    def test_keyword_serialization(self):
        """Test JSON serialization."""
        kw = Keyword(
            text="test",
            score=0.8,
            type="CONCEPT",
            positions=[(0, 4), (10, 14)]
        )
        data = kw.model_dump()
        assert data["text"] == "test"
        assert data["positions"] == [(0, 4), (10, 14)]
    
    def test_keyword_deserialization(self):
        """Test JSON deserialization."""
        data = {
            "text": "test",
            "score": 0.8,
            "type": "CONCEPT"
        }
        kw = Keyword(**data)
        assert kw.text == "test"


class TestExtractionOptions:
    """Test ExtractionOptions model."""
    
    def test_default_options(self):
        """Test default values."""
        opts = ExtractionOptions()
        assert opts.top_k == 10
        assert opts.min_confidence == 0.3
        assert opts.include_entities is True
        assert opts.include_concepts is True
        assert opts.language == "en"
    
    def test_custom_options(self):
        """Test custom values."""
        opts = ExtractionOptions(
            top_k=20,
            min_confidence=0.5,
            language="sv"
        )
        assert opts.top_k == 20
        assert opts.min_confidence == 0.5
        assert opts.language == "sv"
    
    def test_top_k_validation(self):
        """Test top_k bounds."""
        with pytest.raises(ValueError):
            ExtractionOptions(top_k=0)
        
        with pytest.raises(ValueError):
            ExtractionOptions(top_k=101)
    
    def test_language_validation(self):
        """Test language code format."""
        with pytest.raises(ValueError):
            ExtractionOptions(language="english")  # Must be 2-char code


class TestExtractionRequest:
    """Test ExtractionRequest model."""
    
    def test_valid_request(self):
        """Test creating valid request."""
        req = ExtractionRequest(text="Sample text for testing")
        assert req.text == "Sample text for testing"
        assert req.url is None
        assert req.options is None
    
    def test_request_with_options(self):
        """Test request with options."""
        req = ExtractionRequest(
            text="Test",
            url="https://example.com",
            options=ExtractionOptions(top_k=5)
        )
        assert req.options.top_k == 5
        assert req.url == "https://example.com"
    
    def test_empty_text_validation(self):
        """Test empty text is rejected."""
        with pytest.raises(ValueError):
            ExtractionRequest(text="")
        
        with pytest.raises(ValueError):
            ExtractionRequest(text="   ")
    
    def test_text_length_validation(self):
        """Test text length limit."""
        long_text = "x" * 10001
        with pytest.raises(ValueError):
            ExtractionRequest(text=long_text)


class TestExtractionResponse:
    """Test ExtractionResponse model."""
    
    def test_valid_response(self):
        """Test creating valid response."""
        keywords = [
            Keyword(text="test", score=0.9, type="CONCEPT")
        ]
        resp = ExtractionResponse(
            keywords=keywords,
            processing_time=0.123
        )
        assert len(resp.keywords) == 1
        assert resp.processing_time == 0.123
        assert resp.version == "1.0.0"
    
    def test_response_with_metadata(self):
        """Test response with metadata."""
        resp = ExtractionResponse(
            keywords=[],
            processing_time=0.1,
            metadata={"text_length": 100, "candidates": 50}
        )
        assert resp.metadata["text_length"] == 100


class TestBatchExtractionRequest:
    """Test BatchExtractionRequest model."""
    
    def test_valid_batch(self):
        """Test valid batch request."""
        items = [
            ExtractionRequest(text="First document"),
            ExtractionRequest(text="Second document")
        ]
        batch = BatchExtractionRequest(items=items)
        assert len(batch.items) == 2
    
    def test_batch_size_limit(self):
        """Test batch size validation."""
        items = [ExtractionRequest(text=f"Doc {i}") for i in range(101)]
        with pytest.raises(ValueError):
            BatchExtractionRequest(items=items)
    
    def test_empty_batch(self):
        """Test empty batch is rejected."""
        with pytest.raises(ValueError):
            BatchExtractionRequest(items=[])


class TestHealthResponse:
    """Test HealthResponse model."""
    
    def test_healthy_response(self):
        """Test healthy status."""
        health = HealthResponse(
            status="healthy",
            version="1.0.0",
            models_loaded=["model1", "model2"],
            uptime=3600.0
        )
        assert health.status == "healthy"
        assert len(health.models_loaded) == 2
    
    def test_status_validation(self):
        """Test status must be valid."""
        with pytest.raises(ValueError):
            HealthResponse(
                status="invalid",
                version="1.0.0",
                models_loaded=[],
                uptime=0.0
            )


# ============================================================================
# ENGINE TESTS (with mocks)
# ============================================================================

class TestSimpleEngineWithMocks:
    """Test SimpleSemanticEngine with mocked dependencies."""
    
    @patch('sie_x.core.simple_engine.SentenceTransformer', MockSentenceTransformer)
    @patch('sie_x.core.simple_engine.spacy.load', MockSpacy)
    def test_engine_initialization(self):
        """Test engine can be initialized."""
        from sie_x.core.simple_engine import SimpleSemanticEngine
        
        engine = SimpleSemanticEngine()
        assert engine is not None
        assert engine.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    
    @patch('sie_x.core.simple_engine.SentenceTransformer', MockSentenceTransformer)
    @patch('sie_x.core.simple_engine.spacy.load', MockSpacy)
    def test_extract_returns_keywords(self):
        """Test extraction returns keyword list."""
        from sie_x.core.simple_engine import SimpleSemanticEngine
        
        engine = SimpleSemanticEngine()
        keywords = engine.extract("Machine learning and Python are great")
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert all(isinstance(kw, Keyword) for kw in keywords)
    
    @patch('sie_x.core.simple_engine.SentenceTransformer', MockSentenceTransformer)
    @patch('sie_x.core.simple_engine.spacy.load', MockSpacy)
    def test_extract_empty_text(self):
        """Test extraction with empty text."""
        from sie_x.core.simple_engine import SimpleSemanticEngine
        
        engine = SimpleSemanticEngine()
        keywords = engine.extract("")
        
        assert keywords == []
    
    @patch('sie_x.core.simple_engine.SentenceTransformer', MockSentenceTransformer)
    @patch('sie_x.core.simple_engine.spacy.load', MockSpacy)
    def test_extract_respects_top_k(self):
        """Test top_k parameter limits results."""
        from sie_x.core.simple_engine import SimpleSemanticEngine
        
        engine = SimpleSemanticEngine()
        keywords = engine.extract("Test text", top_k=2)
        
        assert len(keywords) <= 2
    
    @patch('sie_x.core.simple_engine.SentenceTransformer', MockSentenceTransformer)
    @patch('sie_x.core.simple_engine.spacy.load', MockSpacy)
    def test_get_stats(self):
        """Test getting engine statistics."""
        from sie_x.core.simple_engine import SimpleSemanticEngine
        
        engine = SimpleSemanticEngine()
        stats = engine.get_stats()
        
        assert "model_name" in stats
        assert "spacy_model" in stats
        assert "cache_size" in stats


# ============================================================================
# MOCK ENGINE TESTS
# ============================================================================

class TestMockEngine:
    """Test the mock engine itself."""
    
    def test_mock_engine_extract(self):
        """Test mock engine returns keywords."""
        engine = MockEngine()
        keywords = engine.extract("Test text")
        
        assert len(keywords) > 0
        assert all(isinstance(kw, Keyword) for kw in keywords)
    
    def test_mock_engine_respects_options(self):
        """Test mock engine respects extraction options."""
        engine = MockEngine()
        
        # Test with entities only
        kw_entities = engine.extract("Test", include_concepts=False)
        assert all(kw.metadata.get("source") == "entity" for kw in kw_entities)
        
        # Test top_k
        kw_limited = engine.extract("Test", top_k=1)
        assert len(kw_limited) <= 1
    
    def test_mock_engine_stats(self):
        """Test mock engine statistics."""
        engine = MockEngine()
        engine.extract("Test")
        
        stats = engine.get_stats()
        assert stats["call_count"] == 1


# ============================================================================
# INTEGRATION PATTERNS (for other components)
# ============================================================================

class TestIntegrationPatterns:
    """Test common integration patterns for API/SDK/CLI."""
    
    def test_api_pattern(self):
        """Test typical API usage pattern."""
        engine = MockEngine()
        
        # Simulate API request
        request = ExtractionRequest(
            text="Sample text about machine learning",
            options=ExtractionOptions(top_k=5)
        )
        
        # Extract keywords
        import time
        start = time.time()
        keywords = engine.extract(
            text=request.text,
            top_k=request.options.top_k
        )
        elapsed = time.time() - start
        
        # Build response
        response = ExtractionResponse(
            keywords=keywords,
            processing_time=elapsed,
            metadata={"text_length": len(request.text)}
        )
        
        assert isinstance(response, ExtractionResponse)
        assert len(response.keywords) <= 5
    
    def test_batch_pattern(self):
        """Test batch processing pattern."""
        engine = MockEngine()
        
        batch = BatchExtractionRequest(
            items=[
                ExtractionRequest(text="First document"),
                ExtractionRequest(text="Second document")
            ],
            options=ExtractionOptions(top_k=3)
        )
        
        # Process batch
        results = []
        for item in batch.items:
            keywords = engine.extract(
                text=item.text,
                top_k=batch.options.top_k if batch.options else 10
            )
            results.append(keywords)
        
        assert len(results) == 2
        assert all(len(kws) <= 3 for kws in results)
    
    def test_health_check_pattern(self):
        """Test health check pattern."""
        engine = MockEngine()
        
        # Simulate health check
        stats = engine.get_stats()
        
        health = HealthResponse(
            status="healthy",
            version="1.0.0",
            models_loaded=[stats["model_name"], stats["spacy_model"]],
            uptime=100.0
        )
        
        assert health.status == "healthy"


# ============================================================================
# FIXTURES FOR OTHER TESTS
# ============================================================================

@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Machine learning is a subset of artificial intelligence that enables
    computers to learn from data. Python is a popular programming language
    for machine learning applications.
    """

@pytest.fixture
def sample_keywords():
    """Sample keywords for testing."""
    return [
        Keyword(text="machine learning", score=0.92, type="CONCEPT"),
        Keyword(text="artificial intelligence", score=0.88, type="CONCEPT"),
        Keyword(text="Python", score=0.85, type="PRODUCT"),
        Keyword(text="data", score=0.75, type="CONCEPT"),
    ]

@pytest.fixture
def mock_engine():
    """Mock engine fixture."""
    return MockEngine()


# ============================================================================
# RUN TESTS
# ============================================================================
# Use pytest from command line: pytest sie_x/core/test_core.py -v
