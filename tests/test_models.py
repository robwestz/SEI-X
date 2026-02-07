"""Tests for Pydantic data models."""

import pytest

from sie_x.core.models import (
    Keyword,
    ExtractionOptions,
    ExtractionRequest,
    ExtractionResponse,
    BatchExtractionRequest,
    HealthResponse,
)


# ============================================================================
# Keyword
# ============================================================================

class TestKeyword:

    def test_creation_with_defaults(self):
        kw = Keyword(text="test", score=0.5, type="CONCEPT")
        assert kw.text == "test"
        assert kw.score == 0.5
        assert kw.count == 1
        assert kw.confidence == 1.0
        assert kw.positions == []
        assert kw.metadata == {}

    def test_score_lower_bound(self):
        with pytest.raises(ValueError):
            Keyword(text="x", score=-0.01, type="CONCEPT")

    def test_score_upper_bound(self):
        with pytest.raises(ValueError):
            Keyword(text="x", score=1.01, type="CONCEPT")

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            Keyword(text="x", score=0.5, type="C", confidence=1.5)

    def test_str_repr(self):
        kw = Keyword(text="ai", score=0.75, type="CONCEPT")
        s = str(kw)
        assert "ai" in s
        assert "0.75" in s

    def test_json_round_trip(self):
        kw = Keyword(
            text="deep learning", score=0.9, type="CONCEPT",
            count=3, confidence=0.85,
            positions=[(0, 13), (50, 63)],
            metadata={"source": "title"},
        )
        data = kw.model_dump()
        kw2 = Keyword(**data)
        assert kw2.text == kw.text
        assert kw2.score == kw.score
        assert kw2.positions == kw.positions
        assert kw2.metadata == kw.metadata


# ============================================================================
# ExtractionOptions
# ============================================================================

class TestExtractionOptions:

    def test_defaults(self):
        opts = ExtractionOptions()
        assert opts.top_k == 10
        assert opts.min_confidence == 0.3
        assert opts.include_entities is True
        assert opts.include_concepts is True
        assert opts.language == "en"

    def test_top_k_too_low(self):
        with pytest.raises(ValueError):
            ExtractionOptions(top_k=0)

    def test_top_k_too_high(self):
        with pytest.raises(ValueError):
            ExtractionOptions(top_k=101)

    def test_language_must_be_two_chars(self):
        with pytest.raises(ValueError):
            ExtractionOptions(language="english")

    def test_valid_language(self):
        opts = ExtractionOptions(language="sv")
        assert opts.language == "sv"


# ============================================================================
# ExtractionRequest
# ============================================================================

class TestExtractionRequest:

    def test_valid_request(self):
        req = ExtractionRequest(text="Hello world")
        assert req.text == "Hello world"
        assert req.url is None
        assert req.options is None

    def test_empty_text_rejected(self):
        with pytest.raises(ValueError):
            ExtractionRequest(text="")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError):
            ExtractionRequest(text="   ")

    def test_max_length(self):
        with pytest.raises(ValueError):
            ExtractionRequest(text="x" * 10001)

    def test_text_at_max_length(self):
        req = ExtractionRequest(text="x" * 10000)
        assert len(req.text) == 10000

    def test_with_options(self):
        req = ExtractionRequest(
            text="test",
            options=ExtractionOptions(top_k=5),
        )
        assert req.options.top_k == 5


# ============================================================================
# ExtractionResponse
# ============================================================================

class TestExtractionResponse:

    def test_valid_response(self):
        resp = ExtractionResponse(keywords=[], processing_time=0.1)
        assert resp.keywords == []
        assert resp.version == "1.0.0"
        assert resp.metadata == {}

    def test_response_with_keywords(self):
        kws = [Keyword(text="ai", score=0.9, type="CONCEPT")]
        resp = ExtractionResponse(keywords=kws, processing_time=0.05)
        assert len(resp.keywords) == 1

    def test_response_with_metadata(self):
        resp = ExtractionResponse(
            keywords=[], processing_time=0.1,
            metadata={"length": 42},
        )
        assert resp.metadata["length"] == 42


# ============================================================================
# BatchExtractionRequest
# ============================================================================

class TestBatchExtractionRequest:

    def test_valid_batch(self):
        items = [
            ExtractionRequest(text="First"),
            ExtractionRequest(text="Second"),
        ]
        batch = BatchExtractionRequest(items=items)
        assert len(batch.items) == 2

    def test_empty_batch_rejected(self):
        with pytest.raises(ValueError):
            BatchExtractionRequest(items=[])

    def test_batch_over_100_rejected(self):
        items = [ExtractionRequest(text=f"doc {i}") for i in range(101)]
        with pytest.raises(ValueError):
            BatchExtractionRequest(items=items)


# ============================================================================
# HealthResponse
# ============================================================================

class TestHealthResponse:

    def test_healthy(self):
        h = HealthResponse(
            status="healthy", version="1.0.0",
            models_loaded=["m1"], uptime=100.0,
        )
        assert h.status == "healthy"

    def test_invalid_status_rejected(self):
        with pytest.raises(ValueError):
            HealthResponse(
                status="broken", version="1.0.0",
                models_loaded=[], uptime=0.0,
            )
