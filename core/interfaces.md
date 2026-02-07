# SIE-X Core Interfaces

This document defines the interfaces between the **core** component and other SIE-X components being developed in parallel.

## Overview

The core component provides foundational data models and the extraction engine. Other components consume these interfaces without depending on implementation details.

## Components in Parallel Development

1. **Core (this component)** - Data models + extraction engine
2. **API Server** - FastAPI server exposing HTTP endpoints
3. **SDK** - Python client library for easy integration
4. **CLI** - Command-line interface for direct usage

## Interface Contracts

### 1. Data Models (Public API)

All components use these Pydantic models from `sie_x.core.models`:

```python
from sie_x.core.models import (
    Keyword,              # Extracted keyword with metadata
    ExtractionOptions,    # Configuration options
    ExtractionRequest,    # Single extraction request
    ExtractionResponse,   # Extraction results
    BatchExtractionRequest,  # Batch processing
    HealthResponse,       # Health check response
)
```

#### Keyword Schema
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

#### ExtractionRequest Schema
```json
{
    "text": "Your document text...",
    "url": "https://example.com/article",
    "options": {
        "top_k": 10,
        "min_confidence": 0.3,
        "include_entities": true,
        "include_concepts": true,
        "language": "en"
    }
}
```

#### ExtractionResponse Schema
```json
{
    "keywords": [{"text": "...", "score": 0.85, ...}],
    "processing_time": 0.234,
    "version": "1.0.0",
    "metadata": {
        "text_length": 1500,
        "candidates": 45
    }
}
```

### 2. Extraction Engine Interface

The extraction engine is accessed through `SimpleSemanticEngine`:

```python
from sie_x.core.simple_engine import SimpleSemanticEngine

# Initialize once (singleton pattern recommended)
engine = SimpleSemanticEngine(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    spacy_model="en_core_web_sm"
)

# Extract keywords
keywords: List[Keyword] = engine.extract(
    text="Your text here...",
    top_k=10,
    min_confidence=0.3,
    include_entities=True,
    include_concepts=True
)

# Get engine statistics
stats = engine.get_stats()
# Returns: {"model_name": "...", "spacy_model": "...", "cache_size": 123}

# Clear cache if needed
engine.clear_cache()
```

#### Method Signature
```python
def extract(
    self,
    text: str,
    top_k: int = 10,
    min_confidence: float = 0.3,
    include_entities: bool = True,
    include_concepts: bool = True
) -> List[Keyword]:
    """
    Extract keywords from text.
    
    Args:
        text: Input text (1-10,000 chars)
        top_k: Max keywords to return (1-100)
        min_confidence: Minimum confidence (0.0-1.0)
        include_entities: Include NER entities
        include_concepts: Include noun phrases
        
    Returns:
        List of Keyword objects, sorted by score
        
    Raises:
        ValueError: If text is empty/invalid
    """
```

### 3. Integration Patterns

#### API Server Integration
```python
# In server.py or minimal_server.py
from fastapi import FastAPI
from sie_x.core.models import ExtractionRequest, ExtractionResponse
from sie_x.core.simple_engine import SimpleSemanticEngine
import time

app = FastAPI()
engine = SimpleSemanticEngine()  # Singleton

@app.post("/extract", response_model=ExtractionResponse)
async def extract_keywords(request: ExtractionRequest):
    start_time = time.time()
    
    # Use options from request or defaults
    options = request.options or ExtractionOptions()
    
    # Extract keywords
    keywords = engine.extract(
        text=request.text,
        top_k=options.top_k,
        min_confidence=options.min_confidence,
        include_entities=options.include_entities,
        include_concepts=options.include_concepts
    )
    
    # Build response
    return ExtractionResponse(
        keywords=keywords,
        processing_time=time.time() - start_time,
        version="1.0.0",
        metadata={
            "text_length": len(request.text),
            "url": request.url
        }
    )
```

#### SDK Integration
```python
# In sdk.py
from sie_x.core.models import (
    ExtractionRequest,
    ExtractionResponse,
    Keyword
)
import httpx

class SieXClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.Client()
    
    def extract(self, text: str, **options) -> ExtractionResponse:
        """Extract keywords via API."""
        request = ExtractionRequest(text=text, options=options)
        response = self.client.post(
            f"{self.base_url}/extract",
            json=request.model_dump()
        )
        return ExtractionResponse(**response.json())
```

#### CLI Integration
```python
# In cli.py
from sie_x.core.simple_engine import SimpleSemanticEngine
from sie_x.core.models import Keyword
import json

def main():
    engine = SimpleSemanticEngine()
    
    # Read from stdin or file
    text = input("Enter text: ")
    
    # Extract
    keywords = engine.extract(text, top_k=10)
    
    # Output as JSON
    output = [kw.model_dump() for kw in keywords]
    print(json.dumps(output, indent=2))
```

## Dependency Flow

```
┌─────────────────────────────────────┐
│   API Server / SDK / CLI            │
│   (Consumers)                       │
└───────────┬─────────────────────────┘
            │
            │ imports models & engine
            ▼
┌─────────────────────────────────────┐
│   sie_x.core                        │
│   ┌─────────────────────────────┐   │
│   │  models.py                  │   │
│   │  - Keyword                  │   │
│   │  - ExtractionRequest/Response│  │
│   │  - Options, Batch, Health   │   │
│   └─────────────────────────────┘   │
│                                     │
│   ┌─────────────────────────────┐   │
│   │  simple_engine.py           │   │
│   │  - SimpleSemanticEngine     │   │
│   │  - extract()                │   │
│   └─────────────────────────────┘   │
└─────────────────────────────────────┘
            │
            │ depends on
            ▼
┌─────────────────────────────────────┐
│   External Libraries                │
│   - spaCy (NER)                     │
│   - sentence-transformers           │
│   - networkx (PageRank)             │
│   - pydantic (validation)           │
└─────────────────────────────────────┘
```

## Testing Interfaces

### Mock Engine (for testing without ML models)

```python
# Use this in tests when you don't want to load heavy models
from sie_x.core.models import Keyword
from typing import List

class MockEngine:
    """Mock extraction engine for testing."""
    
    def extract(
        self,
        text: str,
        top_k: int = 10,
        min_confidence: float = 0.3,
        **kwargs
    ) -> List[Keyword]:
        """Return mock keywords for testing."""
        return [
            Keyword(
                text="test keyword",
                score=0.85,
                type="CONCEPT",
                count=1,
                confidence=0.85
            )
        ]
    
    def get_stats(self):
        return {"model_name": "mock", "cache_size": 0}
    
    def clear_cache(self):
        pass
```

### Test Data Fixtures

```python
# Common test fixtures
SAMPLE_TEXT = """
Machine learning is a subset of artificial intelligence.
It enables computers to learn from data without explicit programming.
"""

EXPECTED_KEYWORDS = [
    "machine learning",
    "artificial intelligence",
    "computers",
    "data",
    "programming"
]
```

## Versioning & Compatibility

### Version Format
- Models: Pydantic v2 with JSON schema
- Engine: Semantic versioning (1.0.0)
- API: Include version in responses

### Breaking Changes
When making changes, ensure:
1. **Models**: Add optional fields, don't remove required fields
2. **Engine**: Keep method signatures backward compatible
3. **API**: Version endpoints if breaking changes needed

### Deprecation Path
```python
# Example: Adding new field (safe)
class Keyword(BaseModel):
    text: str
    score: float
    new_field: Optional[str] = None  # Safe - optional

# Example: Deprecating parameter (safe)
def extract(
    self,
    text: str,
    top_k: int = 10,
    old_param: Optional[int] = None,  # Deprecated but still works
):
    if old_param is not None:
        warnings.warn("old_param is deprecated", DeprecationWarning)
```

## Error Handling

### Expected Exceptions

```python
from pydantic import ValidationError

# Model validation errors
try:
    request = ExtractionRequest(text="")
except ValidationError as e:
    # Handle: text too short/long, invalid options
    pass

# Engine errors
try:
    keywords = engine.extract(text)
except ValueError as e:
    # Handle: empty text, invalid parameters
    pass
except RuntimeError as e:
    # Handle: model loading failures
    pass
```

### Error Response Format (for API)

```json
{
    "error": {
        "type": "ValidationError",
        "message": "Text cannot be empty",
        "details": {...}
    }
}
```

## Performance Considerations

### Engine Initialization
- **Cold start**: ~2-5 seconds (loads ML models)
- **Recommendation**: Initialize once, reuse instance (singleton)

### Extraction Performance
- **Single document**: 50-500ms depending on length
- **Batch processing**: Use `engine.extract()` in parallel
- **Caching**: Embeddings are cached automatically

### Resource Usage
- **Memory**: ~500MB (models) + ~100MB per 1000 cached embeddings
- **CPU**: Heavy during extraction, idle otherwise
- **GPU**: Optional, auto-detected by sentence-transformers

## Configuration

### Environment Variables (recommended)
```bash
# Model selection
SIE_X_SENTENCE_MODEL="sentence-transformers/all-MiniLM-L6-v2"
SIE_X_SPACY_MODEL="en_core_web_sm"

# Performance tuning
SIE_X_CACHE_SIZE=10000
SIE_X_SIMILARITY_THRESHOLD=0.3
```

### Programmatic Configuration
```python
engine = SimpleSemanticEngine(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    spacy_model="en_core_web_sm"
)
```

## Future Extensions

### Planned Interfaces (not yet implemented)
1. **Async extraction**: `async def extract_async(...)`
2. **Streaming**: `def extract_stream(...)` for large documents
3. **Custom models**: Plugin system for domain-specific models
4. **Multi-language**: Language detection and model switching

### Backward Compatibility Promise
- Core models and engine interfaces will remain stable
- New features will be added as optional parameters
- Deprecation notices will be given 2 versions in advance

---

## Quick Reference

### Importing Core Components
```python
# Models
from sie_x.core.models import (
    Keyword,
    ExtractionRequest,
    ExtractionResponse,
    ExtractionOptions,
)

# Engine
from sie_x.core.simple_engine import SimpleSemanticEngine
```

### Minimal Working Example
```python
from sie_x.core.simple_engine import SimpleSemanticEngine

engine = SimpleSemanticEngine()
keywords = engine.extract("Machine learning is amazing!")

for kw in keywords:
    print(f"{kw.text}: {kw.score:.2f}")
```

### Testing Without Dependencies
```python
# Use mock engine in tests
from sie_x.core.test_core import MockEngine

engine = MockEngine()
keywords = engine.extract("test")
assert len(keywords) > 0
```
