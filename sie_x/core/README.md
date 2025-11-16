# SIE-X Core Component

Foundation data models and extraction engine for the Semantic Intelligence Engine X (SIE-X) keyword extraction system.

## Overview

The core component provides:
- **Data Models**: Pydantic models for validation and serialization
- **Extraction Engine**: Semantic keyword extraction using NER + graph ranking
- **Clear Interfaces**: For integration with API, SDK, and CLI components

This is a **self-contained component** designed for parallel development. It can be used independently or integrated with other SIE-X components.

## Architecture

```
sie_x/core/
├── models.py           # Pydantic data models (340+ LOC)
├── simple_engine.py    # Extraction engine (250+ LOC)
├── test_core.py        # Standalone tests with mocks
├── interfaces.md       # Integration documentation
└── README.md          # This file
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.minimal.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Basic Usage

```python
from sie_x.core.simple_engine import SimpleSemanticEngine

# Initialize engine (do this once)
engine = SimpleSemanticEngine()

# Extract keywords
text = """
Machine learning is transforming the technology industry.
Companies use artificial intelligence to analyze data and
make predictions. Python is the most popular programming
language for machine learning applications.
"""

keywords = engine.extract(text, top_k=10, min_confidence=0.3)

# Display results
for kw in keywords:
    print(f"{kw.text:30} | score: {kw.score:.2f} | type: {kw.type}")
```

Output:
```
machine learning               | score: 0.92 | type: CONCEPT
artificial intelligence        | score: 0.88 | type: CONCEPT
Python                        | score: 0.85 | type: PRODUCT
data                          | score: 0.78 | type: CONCEPT
technology industry           | score: 0.75 | type: CONCEPT
```

## Data Models

### Keyword

Represents an extracted keyword with metadata.

```python
from sie_x.core.models import Keyword

keyword = Keyword(
    text="machine learning",
    score=0.92,           # Relevance score (0-1)
    type="CONCEPT",       # Entity type
    count=3,              # Number of occurrences
    confidence=0.88,      # Extraction confidence
    positions=[(10, 27), (45, 62)],  # Character positions
    metadata={"source": "title"}     # Additional data
)

# Serialize to JSON
print(keyword.model_dump_json(indent=2))

# String representation
print(keyword)  # Keyword('machine learning', score=0.92, type=CONCEPT)
```

### ExtractionOptions

Configuration for keyword extraction.

```python
from sie_x.core.models import ExtractionOptions

options = ExtractionOptions(
    top_k=10,                    # Max keywords to return
    min_confidence=0.3,          # Confidence threshold
    include_entities=True,       # Include NER entities
    include_concepts=True,       # Include noun phrases
    language="en"                # Language code
)
```

### ExtractionRequest & Response

Request/response models for API integration.

```python
from sie_x.core.models import ExtractionRequest, ExtractionResponse

# Create request
request = ExtractionRequest(
    text="Your document text...",
    url="https://example.com/article",  # Optional
    options=ExtractionOptions(top_k=5)
)

# Create response
response = ExtractionResponse(
    keywords=[...],
    processing_time=0.234,
    version="1.0.0",
    metadata={"text_length": 1500}
)
```

## Extraction Engine

### SimpleSemanticEngine

The extraction engine combines:
1. **spaCy**: Named Entity Recognition (NER) and noun phrase extraction
2. **Sentence Transformers**: Semantic embeddings for similarity
3. **NetworkX + PageRank**: Graph-based ranking algorithm

### How It Works

```
Input Text
    ↓
1. Candidate Generation (spaCy)
   - Named entities (PERSON, ORG, LOC, etc.)
   - Noun phrases (concepts)
    ↓
2. Semantic Embeddings (Sentence Transformers)
   - Convert to 384-dim vectors
   - Cache for performance
    ↓
3. Similarity Graph (NetworkX)
   - Compute cosine similarity
   - Create edges where similarity > 0.3
    ↓
4. Ranking (PageRank + Frequency)
   - PageRank: 70% weight
   - Frequency: 30% weight
    ↓
5. Filtering & Sorting
   - Apply min_confidence threshold
   - Return top_k results
```

### Advanced Usage

```python
from sie_x.core.simple_engine import SimpleSemanticEngine

# Initialize with custom models
engine = SimpleSemanticEngine(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    spacy_model="en_core_web_sm"
)

# Extract with full options
keywords = engine.extract(
    text=your_text,
    top_k=15,                    # Return top 15 keywords
    min_confidence=0.4,          # Higher threshold
    include_entities=True,       # Include NER
    include_concepts=True        # Include noun phrases
)

# Get engine statistics
stats = engine.get_stats()
print(f"Model: {stats['model_name']}")
print(f"Cache size: {stats['cache_size']}")

# Clear cache if needed
engine.clear_cache()
```

## Use Cases

### 1. Document Summarization

Extract key topics from long documents.

```python
engine = SimpleSemanticEngine()

document = """
[Long document about climate change, renewable energy, 
carbon emissions, solar power, wind turbines, etc.]
"""

keywords = engine.extract(document, top_k=5)
summary = ", ".join(kw.text for kw in keywords)
print(f"Topics: {summary}")
```

### 2. Content Tagging

Automatically tag articles or blog posts.

```python
def tag_article(title, content):
    """Extract tags from article."""
    full_text = f"{title} {title} {content}"  # Weight title higher
    keywords = engine.extract(full_text, top_k=10, min_confidence=0.5)
    
    tags = [kw.text for kw in keywords if kw.type in ["CONCEPT", "ORG", "PRODUCT"]]
    return tags

tags = tag_article(
    "Introduction to Deep Learning",
    "Deep learning is a subset of machine learning..."
)
print(f"Tags: {tags}")
```

### 3. Search Query Expansion

Extract related terms for search enhancement.

```python
def expand_query(query):
    """Expand search query with related keywords."""
    # Use query as context
    context = f"Search for: {query}. Related topics include..."
    
    keywords = engine.extract(context, top_k=5)
    expansion = [kw.text for kw in keywords if kw.score > 0.7]
    
    return [query] + expansion

expanded = expand_query("machine learning")
# Returns: ["machine learning", "artificial intelligence", "neural networks"]
```

### 4. Entity Extraction

Focus on named entities only.

```python
def extract_entities(text):
    """Extract only named entities."""
    keywords = engine.extract(
        text,
        top_k=20,
        include_entities=True,
        include_concepts=False  # Disable concepts
    )
    
    # Group by entity type
    entities = {}
    for kw in keywords:
        if kw.type not in entities:
            entities[kw.type] = []
        entities[kw.type].append(kw.text)
    
    return entities

entities = extract_entities("Apple Inc. announced new products. Tim Cook spoke at the event in Cupertino.")
# Returns: {"ORG": ["Apple Inc."], "PERSON": ["Tim Cook"], "LOC": ["Cupertino"]}
```

### 5. Batch Processing

Process multiple documents efficiently.

```python
from sie_x.core.models import BatchExtractionRequest, ExtractionRequest

documents = [
    "First document about AI...",
    "Second document about blockchain...",
    "Third document about quantum computing..."
]

# Create batch request
batch = BatchExtractionRequest(
    items=[ExtractionRequest(text=doc) for doc in documents],
    options=ExtractionOptions(top_k=5)
)

# Process batch (can be parallelized)
results = []
for item in batch.items:
    keywords = engine.extract(
        text=item.text,
        top_k=batch.options.top_k
    )
    results.append(keywords)

# Display results
for i, keywords in enumerate(results):
    print(f"\nDocument {i+1}:")
    for kw in keywords:
        print(f"  - {kw.text} ({kw.score:.2f})")
```

### 6. Confidence Filtering

Filter results by confidence level.

```python
def get_high_confidence_keywords(text):
    """Extract only high-confidence keywords."""
    keywords = engine.extract(
        text,
        top_k=50,              # Get many candidates
        min_confidence=0.7     # High threshold
    )
    
    # Further filter by score
    top_keywords = [kw for kw in keywords if kw.score > 0.8]
    
    return top_keywords

keywords = get_high_confidence_keywords(text)
```

### 7. Keyword Positions

Track where keywords appear in text.

```python
def find_keyword_positions(text):
    """Find all keyword positions."""
    keywords = engine.extract(text, top_k=20)
    
    for kw in keywords:
        print(f"\n{kw.text} (score: {kw.score:.2f}):")
        print(f"  Occurrences: {kw.count}")
        print(f"  Positions: {kw.positions}")
        
        # Extract context around each position
        for start, end in kw.positions:
            context_start = max(0, start - 20)
            context_end = min(len(text), end + 20)
            context = text[context_start:context_end]
            print(f"  Context: ...{context}...")
```

### 8. Integration with Web Scraping

Extract keywords from web content.

```python
import requests
from bs4 import BeautifulSoup

def extract_from_url(url):
    """Scrape URL and extract keywords."""
    # Fetch content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract text
    text = soup.get_text()
    
    # Extract keywords
    keywords = engine.extract(text, top_k=10)
    
    return keywords

keywords = extract_from_url("https://example.com/article")
```

## Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest sie_x/core/test_core.py -v

# Run specific test class
pytest sie_x/core/test_core.py::TestKeyword -v

# Run with coverage
pytest sie_x/core/test_core.py --cov=sie_x.core
```

### Using Mock Engine

For testing without loading heavy ML models:

```python
from sie_x.core.test_core import MockEngine

# Use in tests
engine = MockEngine()
keywords = engine.extract("test text")

# Mock engine returns predictable results
assert len(keywords) > 0
```

### Test Fixtures

```python
import pytest
from sie_x.core.test_core import sample_text, sample_keywords, mock_engine

def test_my_function(mock_engine, sample_text):
    """Test using fixtures."""
    keywords = mock_engine.extract(sample_text)
    assert len(keywords) > 0
```

## Performance

### Benchmarks

- **Cold start**: ~2-5 seconds (model loading)
- **First extraction**: 200-500ms (no cache)
- **Subsequent extractions**: 50-200ms (with cache)

### Optimization Tips

1. **Initialize once**: Create singleton engine instance
   ```python
   # Good - reuse engine
   engine = SimpleSemanticEngine()
   for text in documents:
       keywords = engine.extract(text)
   
   # Bad - recreate each time
   for text in documents:
       engine = SimpleSemanticEngine()  # Slow!
       keywords = engine.extract(text)
   ```

2. **Batch embeddings**: Process multiple texts together
   ```python
   texts = ["doc1", "doc2", "doc3"]
   # Embeddings are batched internally for efficiency
   ```

3. **Monitor cache**: Check cache size periodically
   ```python
   stats = engine.get_stats()
   if stats['cache_size'] > 10000:
       engine.clear_cache()  # Prevent excessive memory use
   ```

4. **Adjust top_k**: Smaller values are faster
   ```python
   # Fast - only compute top 5
   keywords = engine.extract(text, top_k=5)
   
   # Slower - compute top 50
   keywords = engine.extract(text, top_k=50)
   ```

## Integration with Other Components

### API Server

```python
from fastapi import FastAPI
from sie_x.core.models import ExtractionRequest, ExtractionResponse
from sie_x.core.simple_engine import SimpleSemanticEngine

app = FastAPI()
engine = SimpleSemanticEngine()

@app.post("/extract")
async def extract(request: ExtractionRequest) -> ExtractionResponse:
    keywords = engine.extract(request.text)
    return ExtractionResponse(keywords=keywords, processing_time=0.1)
```

### SDK

```python
from sie_x.core.models import ExtractionRequest

def create_request(text: str) -> ExtractionRequest:
    return ExtractionRequest(text=text)
```

### CLI

```python
import sys
from sie_x.core.simple_engine import SimpleSemanticEngine

engine = SimpleSemanticEngine()
text = sys.stdin.read()
keywords = engine.extract(text)

for kw in keywords:
    print(f"{kw.text}\t{kw.score:.2f}")
```

See [interfaces.md](interfaces.md) for detailed integration documentation.

## Configuration

### Environment Variables

```bash
# Model selection
export SIE_X_SENTENCE_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export SIE_X_SPACY_MODEL="en_core_web_sm"

# Performance tuning
export SIE_X_CACHE_SIZE=10000
export SIE_X_SIMILARITY_THRESHOLD=0.3
```

### Programmatic Configuration

```python
engine = SimpleSemanticEngine(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    spacy_model="en_core_web_md"  # Medium model for better accuracy
)
```

## Troubleshooting

### Model Loading Errors

```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Or use larger model
python -m spacy download en_core_web_md
```

### Memory Issues

```python
# Clear cache periodically
engine.clear_cache()

# Reduce batch size
keywords = engine.extract(text, top_k=5)  # Instead of top_k=50
```

### Import Errors

```python
# Ensure package is in PYTHONPATH
import sys
sys.path.append('/path/to/SEI-X')

from sie_x.core.simple_engine import SimpleSemanticEngine
```

## Limitations

1. **English only**: Currently supports English (en_core_web_sm)
2. **Text length**: Limited to 10,000 characters per request
3. **Cold start**: First initialization takes 2-5 seconds
4. **Memory**: Requires ~500MB for models

## Future Enhancements

- [ ] Multi-language support
- [ ] Async extraction API
- [ ] Custom model plugins
- [ ] Streaming for large documents
- [ ] GPU acceleration support
- [ ] Fine-tuning on domain-specific data

## Dependencies

```
sentence-transformers>=2.2.0   # Semantic embeddings
spacy>=3.5.0                   # NER and linguistic analysis
networkx>=3.0                  # Graph algorithms
pydantic>=2.0.0                # Data validation
numpy>=1.24.0                  # Numerical computing
```

## License

Part of the SIE-X project.

## Contributing

This component is designed for parallel development. When making changes:

1. Keep interfaces stable (see [interfaces.md](interfaces.md))
2. Add tests for new features
3. Update documentation
4. Ensure backward compatibility

## Support

For issues or questions about the core component:
- Check [interfaces.md](interfaces.md) for integration patterns
- See [test_core.py](test_core.py) for usage examples
- Review model docstrings for API details
