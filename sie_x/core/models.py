"""
Core data models for SIE-X keyword extraction system.

These Pydantic models are used throughout the system - engine, API, and SDK.
All models include validation, serialization, and proper documentation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum


class Keyword(BaseModel):
    """
    Represents an extracted keyword with metadata.
    
    Example JSON:
    ```json
    {
        "text": "machine learning",
        "score": 0.85,
        "type": "CONCEPT",
        "count": 3,
        "confidence": 0.92,
        "positions": [[10, 27], [45, 62]],
        "metadata": {"source": "title"}
    }
    ```
    """
    
    text: str = Field(..., description="The keyword text")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    type: str = Field(..., description="Entity type: PERSON, ORG, LOC, CONCEPT, etc.")
    count: int = Field(default=1, ge=1, description="Number of occurrences")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence (0-1)")
    positions: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="Character positions [(start, end), ...]"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Ensure score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be between 0 and 1, got {v}")
        return v
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {v}")
        return v
    
    def __str__(self) -> str:
        """String representation of keyword."""
        return f"Keyword('{self.text}', score={self.score:.2f}, type={self.type})"
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "text": "artificial intelligence",
                "score": 0.89,
                "type": "CONCEPT",
                "count": 2,
                "confidence": 0.95,
                "positions": [[0, 24], [100, 124]],
                "metadata": {"context": "technology"}
            }
        }


class ExtractionOptions(BaseModel):
    """
    Options for keyword extraction.
    
    Example JSON:
    ```json
    {
        "top_k": 10,
        "min_confidence": 0.3,
        "include_entities": true,
        "include_concepts": true,
        "language": "en"
    }
    ```
    """
    
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of keywords to return"
    )
    min_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )
    include_entities: bool = Field(
        default=True,
        description="Include named entities (PERSON, ORG, LOC)"
    )
    include_concepts: bool = Field(
        default=True,
        description="Include concept keywords"
    )
    language: str = Field(
        default="en",
        pattern="^[a-z]{2}$",
        description="ISO 639-1 language code"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "top_k": 15,
                "min_confidence": 0.4,
                "include_entities": True,
                "include_concepts": True,
                "language": "en"
            }
        }


class ExtractionRequest(BaseModel):
    """
    Request for keyword extraction from text or URL.
    
    Example JSON:
    ```json
    {
        "text": "Your document text here...",
        "url": "https://example.com/article",
        "options": {
            "top_k": 10,
            "min_confidence": 0.3
        }
    }
    ```
    """
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to extract keywords from (max 10,000 chars)"
    )
    url: Optional[str] = Field(
        default=None,
        description="Optional source URL for metadata"
    )
    options: Optional[ExtractionOptions] = Field(
        default=None,
        description="Extraction options (uses defaults if not provided)"
    )
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Ensure text is not empty and within limits."""
        v = v.strip()
        if not v:
            raise ValueError("Text cannot be empty")
        if len(v) > 10000:
            raise ValueError(f"Text too long: {len(v)} chars (max 10,000)")
        return v
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "text": "Machine learning is transforming the tech industry...",
                "url": "https://example.com/ml-article",
                "options": {
                    "top_k": 10,
                    "min_confidence": 0.3
                }
            }
        }


class ExtractionResponse(BaseModel):
    """
    Response containing extracted keywords.
    
    Example JSON:
    ```json
    {
        "keywords": [
            {
                "text": "machine learning",
                "score": 0.89,
                "type": "CONCEPT"
            }
        ],
        "processing_time": 0.234,
        "version": "1.0.0",
        "metadata": {
            "text_length": 1500,
            "candidates": 45
        }
    }
    ```
    """
    
    keywords: List[Keyword] = Field(
        ...,
        description="Extracted keywords, sorted by score"
    )
    processing_time: float = Field(
        ...,
        ge=0.0,
        description="Processing time in seconds"
    )
    version: str = Field(
        default="1.0.0",
        description="SIE-X version"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional processing metadata"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "keywords": [
                    {
                        "text": "natural language processing",
                        "score": 0.92,
                        "type": "CONCEPT",
                        "count": 3,
                        "confidence": 0.88
                    }
                ],
                "processing_time": 0.156,
                "version": "1.0.0",
                "metadata": {
                    "text_length": 2500,
                    "candidates_generated": 67
                }
            }
        }


class BatchExtractionRequest(BaseModel):
    """
    Request for batch keyword extraction from multiple texts.
    
    Example JSON:
    ```json
    {
        "items": [
            {"text": "First document..."},
            {"text": "Second document..."}
        ],
        "options": {
            "top_k": 10
        }
    }
    ```
    """
    
    items: List[ExtractionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of extraction requests (max 100)"
    )
    options: Optional[ExtractionOptions] = Field(
        default=None,
        description="Default options for all items (can be overridden per item)"
    )
    
    @field_validator('items')
    @classmethod
    def validate_items(cls, v: List[ExtractionRequest]) -> List[ExtractionRequest]:
        """Ensure batch size is reasonable."""
        if len(v) > 100:
            raise ValueError(f"Too many items: {len(v)} (max 100)")
        return v
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "items": [
                    {"text": "First document about AI and machine learning..."},
                    {"text": "Second document about data science..."}
                ],
                "options": {
                    "top_k": 5,
                    "min_confidence": 0.4
                }
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response for the SIE-X service.
    
    Example JSON:
    ```json
    {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": ["sentence-transformers/all-MiniLM-L6-v2", "en_core_web_sm"],
        "uptime": 3600.5
    }
    ```
    """
    
    status: str = Field(
        ...,
        pattern="^(healthy|degraded|unhealthy)$",
        description="Service status"
    )
    version: str = Field(
        ...,
        description="SIE-X version"
    )
    models_loaded: List[str] = Field(
        ...,
        description="List of loaded ML models"
    )
    uptime: float = Field(
        ...,
        ge=0.0,
        description="Service uptime in seconds"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models_loaded": [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "en_core_web_sm"
                ],
                "uptime": 7200.0
            }
        }
