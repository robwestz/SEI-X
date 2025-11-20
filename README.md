# SIE-X - Semantic Intelligence Engine X

**Production-ready keyword extraction and SEO intelligence platform with multi-language support**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

---

## 🎯 What is SIE-X?

SIE-X is a **semantic intelligence platform** that combines advanced NLP, machine learning, and SEO expertise to extract meaningful insights from text. Built for production use, it powers everything from keyword research to intelligent backlink content generation.

**Core Capabilities:**
- 🔍 **Semantic Keyword Extraction** - PageRank + embeddings + NER
- 🌍 **Multi-Language Support** - 11 languages with auto-detection (including Swedish)
- 🔄 **Streaming Processing** - Handle documents >10K words efficiently
- 🎯 **SEO Intelligence** - Advanced backlink analysis and content optimization
- 🤖 **BACOWR Integration** - Full integration with backlink content writer pipeline
- ⚡ **Production-Ready** - Async API, caching, metrics, Docker support

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/robwestz/SEI-X.git
cd SEI-X

# Install dependencies
pip install -r requirements.minimal.txt

# Download spaCy models (English + Swedish as examples)
python -m spacy download en_core_web_sm
python -m spacy download sv_core_news_sm
```

### Start the API Server

```bash
# Start with uvicorn
uvicorn sie_x.api.minimal_server:app --reload

# Or with Docker
docker-compose -f docker-compose.minimal.yml up
```

**API will be available at:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs (Swagger UI)
- ReDoc: http://localhost:8000/redoc

### Basic Usage

**Python SDK:**
```python
from sie_x.sdk.python.client import SIEXClient

# Initialize client
async with SIEXClient() as client:
    # Extract keywords
    keywords = await client.extract("Your text here", top_k=10)

    # Multi-language extraction (auto-detect)
    keywords = await client.extract_multilang("Hej världen")

    # Streaming for large documents
    async for chunk in client.stream_extract(long_text):
        print(f"Progress: {chunk['progress']}%")
```

**REST API:**
```bash
# Standard extraction
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here", "options": {"top_k": 10}}'

# Multi-language extraction
curl -X POST http://localhost:8000/extract/multilang \
  -d '{"text": "Hej världen, detta är ett exempel"}'

# Streaming extraction (SSE)
curl -N http://localhost:8000/extract/stream \
  -d '{"text": "Very long document..."}'
```

---

## 💡 Key Features

### 1. Advanced Keyword Extraction
- **Sentence Transformers** - Semantic embeddings (all-MiniLM-L6-v2)
- **spaCy NER** - Named entity recognition
- **PageRank Algorithm** - Graph-based ranking (70%) + frequency (30%)
- **Vectorized Operations** - ~10x performance improvement

### 2. Multi-Language Support (11 Languages)
- 🇬🇧 English
- 🇸🇪 Swedish
- 🇪🇸 Spanish
- 🇫🇷 French
- 🇩🇪 German
- 🇮🇹 Italian
- 🇵🇹 Portuguese
- 🇳🇱 Dutch
- 🇬🇷 Greek
- 🇳🇴 Norwegian Bokmål
- 🇱🇹 Lithuanian

**Auto-detection with pattern matching + confidence scores**

### 3. SEO Intelligence

**SEO Transformer:**
- Publisher/target page analysis
- Bridge topic discovery (strong/pivot/wrapper)
- Content brief generation
- Anchor text candidate generation with risk scoring
- Authority signal detection

**BACOWR Integration:**
Full integration with Backlink Content Writer pipeline:
- PageProfiler enrichment
- Intent analyzer augmentation
- Bridge finder (variabelgifte concept)
- BACOWR v2 extensions (intent, links, QC, SERP research)
- Trust policy management (T1-T4)

### 4. Streaming Support
Process large documents efficiently:
- Configurable chunking (1000 words default, 100 word overlap)
- Three merge strategies: union, intersection, weighted
- Server-Sent Events (SSE) for real-time progress
- Memory-efficient async generators

### 5. Production Features
- **Async/Await** - High concurrency support
- **Redis Caching** - Distributed caching with TTL
- **Rate Limiting** - Prevent overload (10 req/sec default)
- **Health Checks** - `/health` endpoint
- **Metrics Collection** - Prometheus-style observability
- **Docker Support** - Ready for containerized deployment
- **CORS Support** - Cross-origin requests enabled

---

## 📚 Documentation

### Core Documentation
- **[HANDOFF.md](HANDOFF.md)** - Complete project overview and implementation details
- **[DEBUG_AND_DEVELOP.md](DEBUG_AND_DEVELOP.md)** - Development roadmap and future vision
- **[sie_x/core/README.md](sie_x/core/README.md)** - Core component usage and troubleshooting
- **[sie_x/core/interfaces.md](sie_x/core/interfaces.md)** - Integration patterns

### API Documentation
- **Swagger UI** - http://localhost:8000/docs
- **ReDoc** - http://localhost:8000/redoc

### Code Documentation
- Comprehensive docstrings on all classes and methods
- Type hints throughout
- Inline comments for complex logic

---

## 🏗️ Architecture

```
SIE-X/
├── sie_x/
│   ├── core/              # Core extraction engine
│   │   ├── models.py      # Pydantic models
│   │   ├── simple_engine.py  # Main extraction engine
│   │   ├── streaming.py   # Streaming support
│   │   └── multilang.py   # Multi-language engine
│   │
│   ├── transformers/      # Domain-specific intelligence
│   │   └── seo_transformer.py  # SEO analysis
│   │
│   ├── integrations/      # External integrations
│   │   └── bacowr_adapter.py   # BACOWR integration
│   │
│   ├── cache/            # Caching layer
│   │   └── redis_cache.py
│   │
│   ├── monitoring/       # Observability
│   │   └── metrics.py
│   │
│   ├── api/              # FastAPI server
│   │   ├── minimal_server.py
│   │   └── routes.py
│   │
│   └── sdk/              # Client SDKs
│       └── python/
│           └── client.py
│
├── demo/                 # Demo applications
│   ├── quickstart.py     # CLI demo
│   └── index.html        # Web UI
│
├── requirements.minimal.txt
├── docker-compose.minimal.yml
└── Dockerfile.minimal
```

---

## 🎨 Use Cases

### 1. Content Creation & SEO
```python
from sie_x.transformers.seo_transformer import SEOTransformer

transformer = SEOTransformer()

# Analyze publisher site
publisher_analysis = await transformer.analyze_publisher(
    text=publisher_content,
    keywords=keywords,
    url="https://publisher.com/article"
)

# Find best bridge topics for natural backlinks
bridges = transformer.find_bridge_topics(
    publisher_analysis,
    target_analysis
)
```

### 2. Multi-Language Content Analysis
```python
from sie_x.core.multilang import MultiLangEngine

engine = MultiLangEngine()

# Auto-detect and extract
keywords = engine.extract("Hej världen, detta är ett exempel")
# Detects Swedish, uses sv_core_news_sm

# Or force specific language
keywords = engine.extract("Some text", language="de")
```

### 3. Large Document Processing
```python
from sie_x.core.streaming import StreamingExtractor

extractor = StreamingExtractor()

# Stream results for 50-page document
async for chunk_result in extractor.extract_stream(long_document):
    print(f"Progress: {chunk_result['progress']}%")
    keywords = chunk_result['keywords']
    # Process keywords incrementally
```

### 4. BACOWR Content Generation
```python
from sie_x.integrations.bacowr_adapter import BACOWRAdapter

adapter = BACOWRAdapter(client)

# Find semantic bridge between publisher and target
bridge = await adapter.find_best_bridge(
    publisher_url="https://publisher.com/article",
    target_url="https://target.com/page",
    serp_context=serp_data
)

# Generate BACOWR v2 extensions
extensions = await adapter.generate_bacowr_extensions(
    bridge=bridge,
    serp_context=serp_data,
    trust_level="T2"  # Academic
)

# Use extensions in content generation
print(extensions['intent_extension']['intent_alignment'])  # 0.85
print(extensions['links_extension']['bridge_type'])        # "strong"
```

---

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW=1.0

# Redis Configuration (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_TTL=3600

# spaCy Configuration
SPACY_MODEL_EN=en_core_web_sm
SPACY_MODEL_SV=sv_core_news_sm
```

### ExtractionOptions

```python
from sie_x.core.models import ExtractionOptions

options = ExtractionOptions(
    top_k=10,              # Number of keywords to return
    min_confidence=0.3,    # Minimum confidence threshold
    include_entities=True, # Include named entities
    include_concepts=True  # Include concept keywords
)
```

### ChunkConfig (Streaming)

```python
from sie_x.core.streaming import ChunkConfig

config = ChunkConfig(
    chunk_size=1000,       # Words per chunk
    overlap=100,           # Overlap between chunks
    min_chunk_size=200     # Minimum chunk size
)
```

---

## 📊 Performance

**Extraction Speed:**
- Single text (1K words): ~50-100ms
- Vectorized similarity: ~10x faster than original
- Cached results: <1ms

**Memory Efficiency:**
- Streaming: O(chunk_size) instead of O(document_size)
- Multi-lang cache: Max 5 models in memory (~2-3GB)
- Engine cache: Configurable (default 100 entries)

**Scalability:**
- Async throughout for high concurrency
- Rate limiting prevents overload
- Redis distributed cache for multi-instance deployment

---

## 🧪 Testing

```bash
# Run Phase 1 tests
pytest sie_x/core/test_core.py -v

# Run with coverage
pytest --cov=sie_x --cov-report=html

# Integration tests (Phase 3 - coming soon)
pytest sie_x/tests/ -v
```

---

## 🛠️ Development

### Local Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Install dev dependencies
pip install pytest pytest-asyncio pytest-mock pytest-cov

# Start development server
uvicorn sie_x.api.minimal_server:app --reload --log-level debug
```

### Code Quality

```bash
# Type checking
mypy sie_x/

# Linting
flake8 sie_x/
pylint sie_x/

# Formatting
black sie_x/
isort sie_x/
```

---

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run
docker-compose -f docker-compose.minimal.yml up -d

# Check logs
docker-compose logs -f sie-x-api

# Scale horizontally
docker-compose up -d --scale sie-x-api=3
```

### Production Considerations

1. **Load Balancing** - Use nginx or HAProxy
2. **Auto-Scaling** - Kubernetes HPA based on CPU/memory
3. **Monitoring** - Prometheus + Grafana dashboards
4. **Logging** - Centralized logging (ELK stack)
5. **Security** - API keys, rate limiting, HTTPS
6. **Caching** - Redis cluster for high availability

---

## 🤝 Contributing

We welcome contributions! See [DEBUG_AND_DEVELOP.md](DEBUG_AND_DEVELOP.md) for development roadmap and ideas.

**Areas for contribution:**
- Additional language support
- New domain-specific transformers
- Performance optimizations
- Integration tests
- Documentation improvements

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **sentence-transformers** - Semantic embeddings
- **spaCy** - NLP and NER
- **FastAPI** - Modern async web framework
- **NetworkX** - Graph algorithms
- **Redis** - Distributed caching

---

## 📞 Support

- **Documentation**: [HANDOFF.md](HANDOFF.md) for complete project overview
- **Issues**: https://github.com/robwestz/SEI-X/issues
- **Discussions**: https://github.com/robwestz/SEI-X/discussions

---

## 🗺️ Roadmap

### Current Status: Phase 2.5 Complete ✅
- [x] Core keyword extraction
- [x] SEO intelligence & BACOWR integration
- [x] Streaming support
- [x] Multi-language (11 languages)
- [x] Production API with caching & metrics

### Phase 3: Integration Tests (Next)
- [ ] SEO Transformer tests
- [ ] BACOWR adapter tests
- [ ] Streaming tests
- [ ] Multi-language tests

### Phase 4+: Advanced Features
- [ ] Additional transformers (Legal, Medical, Financial, etc.)
- [ ] Real-time SERP integration
- [ ] Advanced content optimization
- [ ] Competitor analysis
- [ ] A/B testing framework
- [ ] Analytics dashboard

See [DEBUG_AND_DEVELOP.md](DEBUG_AND_DEVELOP.md) for detailed roadmap and vision.

---

**Built with ❤️ for the SEO and content creation community**
