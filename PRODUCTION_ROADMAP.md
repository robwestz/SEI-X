# SIE-X Production Roadmap v1.0

**Mål:** Färdigställa SIE-X för första kommersiella produktionsversionen
**Deadline:** 3-4 veckor
**Standard:** Enterprise-grade, skalbar, säker, monitorerad

---

## 🎯 Executive Summary

SIE-X har en solid teknisk grund (4,900 LOC, Phase 1-2.5 kompletta) men saknar produktionskritiska komponenter för en kommersiell lansering. Denna roadmap täcker allt som krävs för v1.0.

**Nuvarande state:** Production-ready kod, men untested i verklig miljö
**Målstate:** Fullt testad, monitored, documented, deployable platform

---

## 📋 Phase 3: Kritiska Bugfixar (3-4 dagar)

### Prioritet: 🔴 CRITICAL

Fem kända issues från HANDOFF.md som MÅSTE fixas innan v1.0:

### 3.1 Issue #1: Language Detection Accuracy

**Problem:** Pattern-based detection är basic, kan missta språk (sv/nb, es/pt)
**Current accuracy:** ~75-80%
**Target accuracy:** >95%
**Impact:** Fel språkmodell = dålig keyword extraction

**Solution:**
```python
# sie_x/core/multilang.py - Uppgradera detection

# Current: Pattern-based
def detect_language(text: str) -> tuple[str, float]:
    # Basic word matching...

# New: fasttext integration
import fasttext
model = fasttext.load_model('lid.176.bin')  # Language ID model

def detect_language(text: str) -> tuple[str, float]:
    predictions = model.predict(text, k=2)
    primary_lang = predictions[0][0].replace('__label__', '')
    confidence = float(predictions[1][0])

    # Fallback to pattern-based if low confidence
    if confidence < 0.7:
        return pattern_detect(text)

    return primary_lang, confidence
```

**Tasks:**
- [ ] Install fasttext: `pip install fasttext`
- [ ] Download lid.176.bin model
- [ ] Implement hybrid detection (fasttext + pattern fallback)
- [ ] Add tests for all 11 languages
- [ ] Benchmark accuracy (should be >95%)

**Time:** 4 hours
**Files:** `sie_x/core/multilang.py`, `sie_x/tests/test_multilang.py`

---

### 3.2 Issue #2: BACOWR Compliance - near_window_pass Stub

**Problem:** `near_window_pass` är hardcoded till True (link density inte implementerad)
**Impact:** BACOWR får inkorrekt compliance data
**Risk:** Länkar med för hög density godkänns felaktigt

**Solution:**
```python
# sie_x/integrations/bacowr_adapter.py

def calculate_link_density(content: str, link_positions: List[tuple]) -> float:
    """
    Calculate link density in near-window (150 chars around each link).
    Rule: Max 20% of near-window text can be links.
    """
    total_window_chars = 0
    total_link_chars = 0

    for start, end in link_positions:
        window_start = max(0, start - 75)
        window_end = min(len(content), end + 75)
        window_text = content[window_start:window_end]

        total_window_chars += len(window_text)
        total_link_chars += (end - start)

    if total_window_chars == 0:
        return 0.0

    return total_link_chars / total_window_chars

def check_near_window_pass(content: str, link_positions: List[tuple]) -> bool:
    """Near-window link density must be <= 20%"""
    density = calculate_link_density(content, link_positions)
    return density <= 0.20  # 20% threshold
```

**Tasks:**
- [ ] Implement `calculate_link_density()` function
- [ ] Update `generate_bacowr_extensions()` to use real check
- [ ] Add tests with various link densities (5%, 15%, 25%)
- [ ] Document the 20% threshold in docstrings

**Time:** 3 hours
**Files:** `sie_x/integrations/bacowr_adapter.py`

---

### 3.3 Issue #3: Streaming Fixed Chunk Size

**Problem:** Chunk size är 1000 words oavsett content structure
**Impact:** Chunks kan splittras mitt i paragraf/section
**Better:** Chunka på naturliga boundaries (paragraphs, headers)

**Solution:**
```python
# sie_x/core/streaming.py

def smart_chunk(text: str, target_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Chunk text intelligently based on structure.
    Priority: Headers > Paragraphs > Sentences > Words
    """
    # Detect structure
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_words = len(para.split())

        if current_size + para_words <= target_size:
            current_chunk.append(para)
            current_size += para_words
        else:
            # Finish current chunk
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))

            # Start new chunk with overlap
            if chunks:
                overlap_text = ' '.join(chunks[-1].split()[-overlap:])
                current_chunk = [overlap_text, para]
            else:
                current_chunk = [para]

            current_size = len(para.split())

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks
```

**Tasks:**
- [ ] Implement `smart_chunk()` function
- [ ] Support Markdown headers detection (##, ###)
- [ ] Add tests for different document types (markdown, plain, HTML)
- [ ] Update `ChunkConfig` with `strategy` option

**Time:** 5 hours
**Files:** `sie_x/core/streaming.py`, `sie_x/tests/test_streaming.py`

---

### 3.4 Issue #4: Cache Fallback Distribution

**Problem:** In-memory fallback cache inte distributed (multi-instance problem)
**Impact:** Cache misses vid load balancing med flera instanser
**Solution:** Memcached fallback eller shared Redis cluster

**Solution:**
```python
# sie_x/cache/redis_cache.py

class CacheManager:
    def __init__(self):
        self.redis = None
        self.memcached = None
        self.local = {}  # Last resort

    async def connect(self):
        # Try Redis first
        try:
            self.redis = await aioredis.create_redis_pool(...)
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")

            # Fallback to Memcached
            try:
                import aiomcache
                self.memcached = aiomcache.Client(...)
                logger.info("Fallback to Memcached")
            except Exception as e2:
                logger.warning(f"Memcached unavailable: {e2}, using local cache")

    async def get(self, key: str):
        if self.redis:
            return await self.redis.get(key)
        elif self.memcached:
            return await self.memcached.get(key)
        else:
            return self.local.get(key)

    async def set(self, key: str, value: str, ttl: int):
        if self.redis:
            await self.redis.setex(key, ttl, value)
        elif self.memcached:
            await self.memcached.set(key, value, exptime=ttl)
        else:
            self.local[key] = value
            # Local cache with TTL cleanup
```

**Tasks:**
- [ ] Add Memcached support (`pip install aiomcache`)
- [ ] Implement 3-tier fallback (Redis → Memcached → Local)
- [ ] Add cache health checks
- [ ] Document cache architecture in README

**Time:** 6 hours
**Files:** `sie_x/cache/redis_cache.py`

---

### 3.5 Issue #5: spaCy Model Auto-Download

**Problem:** Users måste manuellt ladda ner alla spaCy models
**Impact:** Dålig UX, fel meddelanden, confusion
**Solution:** Auto-download on first use med progress bar

**Solution:**
```python
# sie_x/core/multilang.py

import subprocess
import sys
from tqdm import tqdm

def ensure_spacy_model(model_name: str) -> None:
    """
    Ensure spaCy model is installed, download if missing.
    Shows progress bar during download.
    """
    try:
        spacy.load(model_name)
        logger.info(f"Model {model_name} already installed")
    except OSError:
        logger.info(f"Downloading spaCy model: {model_name}")

        # Download with progress
        print(f"📦 Installing {model_name}...")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", model_name,
            "--quiet"
        ])

        logger.info(f"Model {model_name} installed successfully")

class MultiLangEngine:
    def __init__(self):
        self.engines = {}

    def _load_engine(self, lang: str):
        model_name = self.LANG_MODELS.get(lang)

        # Auto-download if missing
        ensure_spacy_model(model_name)

        # Now load normally
        nlp = spacy.load(model_name)
        ...
```

**Tasks:**
- [ ] Implement `ensure_spacy_model()` function
- [ ] Add progress indicators (tqdm or custom)
- [ ] Handle download failures gracefully
- [ ] Add option to disable auto-download (env var)
- [ ] Document in README

**Time:** 3 hours
**Files:** `sie_x/core/multilang.py`, `README.md`

---

## 📋 Phase 4: Integration Testing (5-7 dagar)

### Prioritet: 🔴 CRITICAL

**Problem:** Phase 2 och 2.5 komponenter saknar integration tests
**Impact:** Kan inte garantera system fungerar end-to-end i production

### 4.1 Test Infrastructure Setup

**Tasks:**
- [ ] Create `sie_x/tests/` directory structure
- [ ] Setup pytest fixtures för engines, clients, mock data
- [ ] Configure pytest-asyncio för async tests
- [ ] Setup test Redis instance (Docker)
- [ ] Create test coverage reporting

**Files to create:**
```
sie_x/tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_seo_transformer.py
├── test_bacowr_adapter.py
├── test_streaming.py
├── test_multilang.py
├── test_cache_integration.py
├── test_metrics.py
└── test_api_end_to_end.py
```

**Time:** 1 dag

---

### 4.2 SEO Transformer Tests

**Coverage:**
- [ ] `analyze_publisher()` - publisher analysis accuracy
- [ ] `analyze_target()` - target analysis accuracy
- [ ] `find_bridge_topics()` - bridge discovery (strong/pivot/wrapper)
- [ ] `generate_content_brief()` - content brief generation

**Test cases:**
```python
# sie_x/tests/test_seo_transformer.py

@pytest.mark.asyncio
async def test_publisher_analysis_complete(transformer, sample_publisher_text):
    """Test publisher analysis returns all required fields"""
    result = await transformer.analyze_publisher(
        text=sample_publisher_text,
        keywords=[...],
        url="https://example.com"
    )

    assert 'entities' in result
    assert 'topics' in result
    assert 'content_type' in result
    assert 'authority_signals' in result
    assert len(result['topics']) >= 3

@pytest.mark.asyncio
async def test_bridge_discovery_strong(transformer, pub_analysis, tgt_analysis):
    """Test strong bridge discovery (>0.8 similarity)"""
    bridges = transformer.find_bridge_topics(pub_analysis, tgt_analysis)

    strong_bridges = [b for b in bridges if b.bridge_type == 'strong']
    assert len(strong_bridges) > 0
    assert all(b.strength >= 0.8 for b in strong_bridges)

@pytest.mark.asyncio
async def test_content_brief_generation(transformer, bridge):
    """Test content brief has all required fields"""
    brief = await transformer.generate_content_brief(
        bridge=bridge,
        pub_analysis={...},
        tgt_analysis={...}
    )

    assert 'primary_topic' in brief
    assert 'semantic_keywords' in brief
    assert 'must_mention_entities' in brief
    assert 'tone_alignment' in brief
    assert brief['recommended_length'] > 0
```

**Time:** 1 dag
**Target coverage:** >80%

---

### 4.3 BACOWR Adapter Tests

**Coverage:**
- [ ] `enhance_page_profiler()` - PageProfiler enrichment
- [ ] `find_best_bridge()` - Bridge finder algorithm
- [ ] `generate_bacowr_extensions()` - BACOWR v2 schema compliance
- [ ] Trust policy logic (T1-T4)
- [ ] Compliance checks (LSI, near_window, publisher_fit, anchor_risk)
- [ ] Variabelgifte scoring

**Critical test:**
```python
# sie_x/tests/test_bacowr_adapter.py

@pytest.mark.asyncio
async def test_bacowr_extensions_schema_compliance():
    """
    Test BACOWR v2 extensions schema EXACT compliance.
    This is CRITICAL - schema mismatch breaks BACOWR pipeline.
    """
    adapter = BACOWRAdapter(client)
    extensions = await adapter.generate_bacowr_extensions(
        bridge=mock_bridge,
        serp_context=mock_serp,
        trust_level="T2"
    )

    # Validate required top-level keys
    assert 'intent_extension' in extensions
    assert 'links_extension' in extensions
    assert 'qc_extension' in extensions

    # Validate intent_extension schema
    assert 'serp_intent_primary' in extensions['intent_extension']
    assert extensions['intent_extension']['serp_intent_primary'] in [
        'informational', 'commercial', 'navigational', 'transactional'
    ]
    assert 'intent_alignment' in extensions['intent_extension']
    assert 0 <= extensions['intent_extension']['intent_alignment'] <= 1

    # Validate links_extension schema
    links = extensions['links_extension']
    assert links['bridge_type'] in ['strong', 'pivot', 'wrapper']
    assert 'trust_policy' in links
    assert all(k in links['trust_policy'] for k in ['do_follow', 'sponsored', 'ugc'])
    assert 'compliance' in links
    assert all(k in links['compliance'] for k in [
        'lsi_quality_pass', 'near_window_pass', 'publisher_fit_pass', 'anchor_risk_acceptable'
    ])

    # Validate QC extension
    qc = extensions['qc_extension']
    assert qc['anchor_risk'] in ['low', 'medium', 'high']
    assert 'readability' in qc
    assert qc['thresholds_version'] == 'v2.1'

@pytest.mark.asyncio
async def test_trust_policy_by_level():
    """Test trust policy T1-T4 is correctly assigned"""
    test_cases = [
        ('T1', {'do_follow': True, 'sponsored': False, 'ugc': False}),
        ('T2', {'do_follow': True, 'sponsored': False, 'ugc': False}),
        ('T3', {'do_follow': True, 'sponsored': False, 'ugc': False}),
        ('T4', {'do_follow': True, 'sponsored': True, 'ugc': False}),
    ]

    for trust_level, expected_policy in test_cases:
        extensions = await adapter.generate_bacowr_extensions(
            bridge=mock_bridge,
            serp_context=mock_serp,
            trust_level=trust_level
        )

        assert extensions['links_extension']['trust_policy'] == expected_policy

@pytest.mark.asyncio
async def test_near_window_compliance_real_check():
    """
    Test near_window_pass uses REAL link density calculation.
    This was Issue #2 - must verify it's fixed.
    """
    # Mock content with high link density (should fail)
    high_density_content = "Some text [link1 long anchor text here] more text [link2]"
    link_positions = [(10, 40), (50, 56)]  # 46 chars of links in ~60 char content

    extensions = await adapter.generate_bacowr_extensions(
        bridge=mock_bridge,
        serp_context=mock_serp,
        trust_level="T1",
        content=high_density_content,
        link_positions=link_positions
    )

    # Should fail near_window check (density > 20%)
    assert extensions['links_extension']['compliance']['near_window_pass'] == False

    # Now test with low density (should pass)
    low_density_content = "Lots of text here. " * 20 + "[link]"
    link_positions = [(400, 406)]  # Small link in large text

    extensions = await adapter.generate_bacowr_extensions(
        bridge=mock_bridge,
        serp_context=mock_serp,
        trust_level="T1",
        content=low_density_content,
        link_positions=link_positions
    )

    assert extensions['links_extension']['compliance']['near_window_pass'] == True
```

**Time:** 1.5 dagar
**Target coverage:** >85% (BACOWR integration är critical)

---

### 4.4 Streaming Tests

**Coverage:**
- [ ] Chunk splitting logic (word boundaries, overlap)
- [ ] Smart chunking (paragraphs, headers)
- [ ] Three merge strategies (union, intersection, weighted)
- [ ] SSE streaming (Server-Sent Events)
- [ ] Large document handling (>10K words)
- [ ] Progress tracking accuracy

**Test cases:**
```python
# sie_x/tests/test_streaming.py

@pytest.mark.asyncio
async def test_smart_chunking_respects_paragraphs():
    """Test chunks break on paragraph boundaries"""
    text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
    chunks = smart_chunk(text, target_size=50)

    # No chunk should split mid-paragraph
    for chunk in chunks:
        assert '\n\n' not in chunk or chunk.count('\n\n') >= 1

@pytest.mark.asyncio
async def test_merge_strategy_weighted():
    """Test weighted merge gives correct scores"""
    chunk_results = [
        [Keyword(text="AI", score=0.9, count=3)],
        [Keyword(text="AI", score=0.8, count=2)],
        [Keyword(text="ML", score=0.7, count=1)]
    ]

    merged = merge_keywords(chunk_results, strategy="weighted")

    # "AI" appears in 2 chunks with higher frequency → higher score
    ai_keyword = next(k for k in merged if k.text == "AI")
    ml_keyword = next(k for k in merged if k.text == "ML")

    assert ai_keyword.score > ml_keyword.score
    assert ai_keyword.count == 5  # 3 + 2

@pytest.mark.asyncio
async def test_sse_streaming_format():
    """Test Server-Sent Events format is correct"""
    response = client.post("/extract/stream", json={"text": long_text})

    events = []
    async for line in response.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    # Should have chunk events + final event
    assert len(events) >= 2
    assert events[-1]['type'] == 'final'
    assert 'progress' in events[0]
    assert 0 <= events[0]['progress'] <= 100
```

**Time:** 1 dag
**Target coverage:** >75%

---

### 4.5 Multi-Language Tests

**Coverage:**
- [ ] Language detection accuracy for all 11 languages
- [ ] Model loading for each language
- [ ] Cache eviction (FIFO) tests
- [ ] Fallback to English when detection fails
- [ ] API `/languages` endpoint
- [ ] Mixed-language text handling

**Test cases:**
```python
# sie_x/tests/test_multilang.py

@pytest.mark.parametrize("lang,sample_text,expected", [
    ('en', "Hello world", 'en'),
    ('sv', "Hej världen", 'sv'),
    ('es', "Hola mundo", 'es'),
    ('fr', "Bonjour monde", 'fr'),
    ('de', "Hallo Welt", 'de'),
    # ... all 11 languages
])
def test_language_detection_accuracy(lang, sample_text, expected):
    """Test detection accuracy for all supported languages"""
    engine = MultiLangEngine()
    detected, confidence = engine.detect_language(sample_text)

    assert detected == expected
    assert confidence > 0.7  # Minimum confidence threshold

@pytest.mark.asyncio
async def test_model_loading_all_languages():
    """Test all 11 language models can be loaded"""
    engine = MultiLangEngine()

    for lang in engine.SUPPORTED_LANGUAGES:
        try:
            keywords = engine.extract(
                text=f"Sample text in {lang}",
                language=lang
            )
            assert len(keywords) >= 0  # Should not crash
        except Exception as e:
            pytest.fail(f"Failed to load {lang}: {e}")

@pytest.mark.asyncio
async def test_cache_eviction_fifo():
    """Test FIFO cache evicts oldest language when max reached"""
    engine = MultiLangEngine(max_cache=3)

    # Load 3 languages
    engine.extract("Hello", language="en")
    engine.extract("Hola", language="es")
    engine.extract("Bonjour", language="fr")

    assert len(engine.engines) == 3

    # Load 4th language - should evict oldest (en)
    engine.extract("Hallo", language="de")

    assert len(engine.engines) == 3
    assert 'en' not in engine.engines
    assert 'de' in engine.engines

@pytest.mark.asyncio
async def test_fasttext_detection_upgrade():
    """
    Test fasttext integration (Issue #1 fix).
    Should have >95% accuracy.
    """
    test_samples = [
        # Add 100+ real-world samples in each language
        ...
    ]

    correct = 0
    total = len(test_samples)

    for expected_lang, text in test_samples:
        detected, confidence = engine.detect_language(text)
        if detected == expected_lang:
            correct += 1

    accuracy = correct / total
    assert accuracy > 0.95  # >95% target accuracy
```

**Time:** 1 dag
**Target coverage:** >80%

---

### 4.6 Cache Integration Tests

**Coverage:**
- [ ] Redis connection and operations
- [ ] Fallback to Memcached when Redis fails
- [ ] Fallback to local cache when both fail
- [ ] TTL expiration
- [ ] Cache invalidation
- [ ] Concurrent access

**Time:** 0.5 dag
**Target coverage:** >75%

---

### 4.7 Metrics Tests

**Coverage:**
- [ ] Counter increments
- [ ] Histogram recording (latency percentiles)
- [ ] Gauge updates
- [ ] Timer context manager
- [ ] Prometheus export format

**Time:** 0.5 dag
**Target coverage:** >70%

---

### 4.8 API End-to-End Tests

**Coverage:**
- [ ] All endpoints (`/extract`, `/extract/batch`, `/extract/stream`, `/extract/multilang`)
- [ ] Authentication (when implemented)
- [ ] Rate limiting triggers
- [ ] Error responses (400, 429, 500)
- [ ] CORS headers
- [ ] Health checks
- [ ] Concurrent requests (load simulation)

**Test cases:**
```python
# sie_x/tests/test_api_end_to_end.py

@pytest.mark.asyncio
async def test_full_extraction_pipeline(client):
    """Test complete extraction flow end-to-end"""
    response = await client.post("/extract", json={
        "text": "Machine learning is transforming technology.",
        "options": {"top_k": 5, "min_confidence": 0.3}
    })

    assert response.status_code == 200
    data = response.json()

    assert 'keywords' in data
    assert 'processing_time' in data
    assert len(data['keywords']) <= 5
    assert all(k['confidence'] >= 0.3 for k in data['keywords'])

@pytest.mark.asyncio
async def test_rate_limiting_enforcement(client):
    """Test rate limiter blocks excess requests"""
    # Send 15 requests rapidly (limit is 10/sec)
    responses = []
    for _ in range(15):
        resp = await client.post("/extract", json={"text": "test"})
        responses.append(resp)

    # Some should be rate limited
    status_codes = [r.status_code for r in responses]
    assert 429 in status_codes  # Too Many Requests

@pytest.mark.asyncio
async def test_concurrent_requests_scaling(client):
    """Test API handles concurrent load"""
    import asyncio

    async def make_request():
        return await client.post("/extract", json={"text": "test text"})

    # 50 concurrent requests
    tasks = [make_request() for _ in range(50)]
    responses = await asyncio.gather(*tasks)

    # All should succeed (assuming no rate limit hit)
    assert all(r.status_code == 200 for r in responses)
```

**Time:** 1 dag
**Target coverage:** Full API surface

---

**Phase 4 Total:** 5-7 dagar
**Target overall coverage:** >80%

---

## 📋 Phase 5: Security Audit (2-3 dagar)

### Prioritet: 🟡 HIGH

**Requirement:** Kommersiell produkt måste vara säker mot common vulnerabilities

### 5.1 Input Validation & Injection Prevention

**Threats:**
- XSS (Cross-Site Scripting)
- SQL Injection (inte applicable, ingen SQL)
- Command Injection
- Path Traversal
- NoSQL Injection (Redis)

**Tasks:**
- [ ] Sanitize all user inputs (text, URLs, options)
- [ ] Validate URL schemes (only http/https)
- [ ] Escape special characters in Redis keys
- [ ] Set max text length (current: 10K chars, enforce strictly)
- [ ] Validate JSON payloads (Pydantic should handle this)

**Implementation:**
```python
# sie_x/api/minimal_server.py

from fastapi import HTTPException
import re

MAX_TEXT_LENGTH = 10_000
ALLOWED_URL_SCHEMES = ['http', 'https']

def validate_text_input(text: str) -> str:
    """Validate and sanitize text input"""
    if not text or not text.strip():
        raise HTTPException(400, "Text cannot be empty")

    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(400, f"Text exceeds maximum length ({MAX_TEXT_LENGTH} chars)")

    # Remove null bytes (potential injection)
    text = text.replace('\x00', '')

    return text.strip()

def validate_url_input(url: str) -> str:
    """Validate URL input"""
    if not url:
        return url

    from urllib.parse import urlparse
    parsed = urlparse(url)

    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        raise HTTPException(400, f"Invalid URL scheme. Allowed: {ALLOWED_URL_SCHEMES}")

    # Prevent SSRF (Server-Side Request Forgery)
    if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
        raise HTTPException(400, "Cannot access local URLs")

    return url

@app.post("/extract")
async def extract(request: ExtractionRequest):
    # Validate inputs
    request.text = validate_text_input(request.text)
    if request.url:
        request.url = validate_url_input(request.url)

    # Continue processing...
```

**Time:** 1 dag

---

### 5.2 API Authentication & Authorization

**Current:** Ingen authentication (alla requests godkänns)
**Requirement:** API keys för production, JWT för user sessions

**Implementation:**
```python
# sie_x/api/auth.py

from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

security = HTTPBearer()

# In production, load from env or database
API_KEYS = {
    "sk_test_abc123": {"user_id": "user_1", "tier": "free", "rate_limit": 100},
    "sk_prod_xyz789": {"user_id": "user_2", "tier": "pro", "rate_limit": 10000},
}

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key from Authorization header"""
    token = credentials.credentials

    if token.startswith("sk_"):
        # API key authentication
        if token not in API_KEYS:
            raise HTTPException(401, "Invalid API key")

        return API_KEYS[token]

    else:
        # JWT authentication
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(401, "Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(401, "Invalid token")

@app.post("/extract")
async def extract(
    request: ExtractionRequest,
    auth = Depends(verify_api_key)
):
    user_id = auth['user_id']

    # Track usage per user
    await track_usage(user_id, endpoint="extract")

    # Continue processing...
```

**Tasks:**
- [ ] Implement API key verification
- [ ] Implement JWT token verification
- [ ] Add user registration/login endpoints
- [ ] Store API keys securely (hashed in database)
- [ ] Add usage tracking per user
- [ ] Implement tier-based rate limiting

**Time:** 1 dag

---

### 5.3 Rate Limiting Per User

**Current:** IP-based rate limiting (10 req/sec)
**Problem:** Shared IPs (NAT, proxies) punish all users
**Better:** Per-user/API-key rate limiting

**Implementation:**
```python
# sie_x/api/minimal_server.py

from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

class UserRateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)  # user_id -> [timestamps]
        self.limits = {
            "free": (100, 3600),      # 100 requests per hour
            "pro": (10000, 3600),     # 10K requests per hour
            "enterprise": (None, None) # Unlimited
        }

    async def check_limit(self, user_id: str, tier: str) -> bool:
        """Check if user is within rate limit"""
        if tier == "enterprise":
            return True

        max_requests, window_seconds = self.limits[tier]
        now = datetime.now()
        cutoff = now - timedelta(seconds=window_seconds)

        # Remove old requests
        self.requests[user_id] = [
            ts for ts in self.requests[user_id]
            if ts > cutoff
        ]

        if len(self.requests[user_id]) >= max_requests:
            return False

        self.requests[user_id].append(now)
        return True

rate_limiter = UserRateLimiter()

@app.post("/extract")
async def extract(request: ExtractionRequest, auth = Depends(verify_api_key)):
    user_id = auth['user_id']
    tier = auth['tier']

    if not await rate_limiter.check_limit(user_id, tier):
        raise HTTPException(
            429,
            f"Rate limit exceeded for {tier} tier. Upgrade for higher limits."
        )

    # Continue processing...
```

**Time:** 0.5 dag

---

### 5.4 Data Privacy & GDPR Compliance

**Requirements:**
- No PII stored without consent
- Data deletion on request
- Data export on request
- Audit logs

**Tasks:**
- [ ] Document what data is collected (usage logs, cached keywords)
- [ ] Implement data retention policy (auto-delete after 90 days)
- [ ] Add user data export endpoint
- [ ] Add user data deletion endpoint
- [ ] Create Privacy Policy and Terms of Service
- [ ] Add cookie consent (if web UI is used)

**Implementation:**
```python
# sie_x/api/privacy.py

@app.get("/user/data/export")
async def export_user_data(auth = Depends(verify_api_key)):
    """Export all user data (GDPR Article 15)"""
    user_id = auth['user_id']

    data = {
        'user_id': user_id,
        'usage_logs': await db.get_usage_logs(user_id),
        'cached_extractions': await db.get_cached_extractions(user_id),
        'api_keys': await db.get_api_keys(user_id),
    }

    return JSONResponse(content=data)

@app.delete("/user/data")
async def delete_user_data(auth = Depends(verify_api_key)):
    """Delete all user data (GDPR Article 17 - Right to be forgotten)"""
    user_id = auth['user_id']

    await db.delete_usage_logs(user_id)
    await db.delete_cached_extractions(user_id)
    await db.revoke_api_keys(user_id)

    return {"status": "deleted"}
```

**Time:** 0.5 dag (implementation + legal docs)

---

**Phase 5 Total:** 2-3 dagar

---

## 📋 Phase 6: Performance Optimization (2-3 dagar)

### Prioritet: 🟢 MEDIUM

**Current performance:** 50-100ms per extraction
**Target:** <30ms with caching, <50ms cold

### 6.1 Load Testing & Benchmarking

**Tools:**
- Locust (Python load testing)
- Apache Bench (ab)
- wrk (modern HTTP benchmarking)

**Tasks:**
- [ ] Setup Locust test scenarios
- [ ] Test scenarios:
  - 100 concurrent users
  - 1000 requests/sec sustained
  - Spike test (0 → 1000 users in 10 sec)
  - Soak test (24 hour continuous load)
- [ ] Identify bottlenecks (CPU, memory, I/O, database)
- [ ] Create performance baseline report

**Implementation:**
```python
# tests/load/locustfile.py

from locust import HttpUser, task, between

class SIEXUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def extract_short(self):
        """Most common: short text extraction"""
        self.client.post("/extract", json={
            "text": "Machine learning is transforming technology.",
            "options": {"top_k": 10}
        })

    @task(1)
    def extract_long(self):
        """Less common: long text extraction"""
        self.client.post("/extract", json={
            "text": "Long article text..." * 100,
            "options": {"top_k": 20}
        })

    @task(1)
    def multilang(self):
        """Multi-language detection"""
        self.client.post("/extract/multilang", json={
            "text": "Hej världen, detta är ett exempel"
        })

# Run: locust -f tests/load/locustfile.py --host=http://localhost:8000
```

**Time:** 1 dag

---

### 6.2 Optimization Implementation

**Findings from load test will determine priorities. Common optimizations:**

**A. Model Quantization**
- Reduce sentence-transformer model size (384-dim → 128-dim)
- Use quantized spaCy models
- Trade-off: ~5% accuracy loss, 3x speedup

**B. Batch Processing**
- Process multiple requests together
- Batch embedding generation
- Amortize model overhead

**C. Caching Strategy**
- Cache embeddings by text hash
- Cache entire extraction results
- Use Redis pipelining

**D. Async Optimizations**
- Connection pooling (Redis, HTTP clients)
- Parallel processing where possible
- Reduce await overhead

**E. Code-level optimizations**
- Profile with cProfile
- Optimize hot paths (vectorized operations already done)
- Remove unnecessary computations

**Time:** 1-2 dagar (based on findings)

---

**Phase 6 Total:** 2-3 dagar

---

## 📋 Phase 7: Production Deployment (3-4 dagar)

### Prioritet: 🟡 HIGH

### 7.1 Kubernetes Deployment

**Tasks:**
- [ ] Create Kubernetes manifests (Deployment, Service, Ingress)
- [ ] Configure Horizontal Pod Autoscaler (HPA)
- [ ] Setup Persistent Volume Claims (for model cache)
- [ ] Configure liveness and readiness probes
- [ ] Setup secrets management (API keys, JWT secret)
- [ ] Create staging and production namespaces

**Files to create:**
```
k8s/
├── base/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── secrets.yaml
├── overlays/
│   ├── staging/
│   │   └── kustomization.yaml
│   └── production/
│       └── kustomization.yaml
└── hpa.yaml
```

**Example deployment.yaml:**
```yaml
# k8s/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sie-x-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sie-x-api
  template:
    metadata:
      labels:
        app: sie-x-api
    spec:
      containers:
      - name: api
        image: sie-x:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: redis-service
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: sie-x-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sie-x-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sie-x-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Time:** 2 dagar

---

### 7.2 CI/CD Pipeline

**Platform:** GitHub Actions

**Tasks:**
- [ ] Create `.github/workflows/ci.yml` - Tests on every PR
- [ ] Create `.github/workflows/cd.yml` - Deploy on merge to main
- [ ] Setup Docker image building and pushing
- [ ] Configure automatic testing (pytest)
- [ ] Setup code quality checks (flake8, mypy, black)
- [ ] Configure deployment to staging/production
- [ ] Add manual approval for production deploys

**Example CI workflow:**
```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
  push:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.minimal.txt
        pip install pytest pytest-asyncio pytest-cov
        python -m spacy download en_core_web_sm

    - name: Run tests
      run: |
        pytest sie_x/tests/ -v --cov=sie_x --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml

  lint:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install linters
      run: pip install flake8 mypy black isort

    - name: Run linters
      run: |
        flake8 sie_x/
        mypy sie_x/
        black --check sie_x/
        isort --check sie_x/

  docker-build:
    runs-on: ubuntu-latest
    needs: [test, lint]

    steps:
    - uses: actions/checkout@v3

    - name: Build Docker image
      run: docker build -f Dockerfile.minimal -t sie-x:${{ github.sha }} .

    - name: Test Docker image
      run: |
        docker run -d -p 8000:8000 sie-x:${{ github.sha }}
        sleep 10
        curl http://localhost:8000/health
```

**CD workflow:**
```yaml
# .github/workflows/cd.yml
name: CD

on:
  push:
    branches: [main]

jobs:
  deploy-staging:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Build and push Docker image
      env:
        DOCKER_USERNAME: ${{ secrets.DOCKER_USERNAME }}
        DOCKER_PASSWORD: ${{ secrets.DOCKER_PASSWORD }}
      run: |
        echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin
        docker build -f Dockerfile.minimal -t sie-x:${{ github.sha }} .
        docker tag sie-x:${{ github.sha }} sie-x:staging
        docker push sie-x:staging

    - name: Deploy to staging
      run: |
        kubectl set image deployment/sie-x-api api=sie-x:staging -n staging
        kubectl rollout status deployment/sie-x-api -n staging

  deploy-production:
    runs-on: ubuntu-latest
    needs: deploy-staging
    environment: production  # Requires manual approval

    steps:
    - name: Deploy to production
      run: |
        docker tag sie-x:staging sie-x:production
        docker push sie-x:production
        kubectl set image deployment/sie-x-api api=sie-x:production -n production
        kubectl rollout status deployment/sie-x-api -n production
```

**Time:** 1 dag

---

### 7.3 Monitoring & Alerting

**Stack:** Prometheus + Grafana + AlertManager

**Tasks:**
- [ ] Setup Prometheus metrics collection
- [ ] Create Grafana dashboards
- [ ] Configure AlertManager rules
- [ ] Setup notification channels (Slack, email, PagerDuty)
- [ ] Define SLIs/SLOs (Service Level Indicators/Objectives)

**SLIs/SLOs:**
```yaml
# SLO definitions
availability:
  target: 99.9%  # 3 nines
  measurement: successful_requests / total_requests

latency:
  p50: < 50ms
  p95: < 100ms
  p99: < 200ms

error_rate:
  target: < 0.1%  # Less than 1 error per 1000 requests
```

**Grafana Dashboard metrics:**
- Request rate (req/sec)
- Error rate (errors/sec)
- Latency percentiles (p50, p95, p99)
- Cache hit ratio
- Active connections
- Memory usage
- CPU usage
- Pod count (autoscaling)

**AlertManager rules:**
```yaml
# alertmanager/rules.yml
groups:
- name: sie-x
  interval: 30s
  rules:
  - alert: HighErrorRate
    expr: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.01
    for: 5m
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }}%"

  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.2
    for: 5m
    annotations:
      summary: "High latency detected"
      description: "P95 latency is {{ $value }}s"

  - alert: LowCacheHitRate
    expr: rate(cache_hits[5m]) / rate(cache_requests[5m]) < 0.5
    for: 10m
    annotations:
      summary: "Cache hit rate below 50%"
```

**Time:** 1 dag

---

### 7.4 Documentation for Deployment

**Tasks:**
- [ ] Create DEPLOYMENT.md with step-by-step guide
- [ ] Document environment variables
- [ ] Document scaling procedures
- [ ] Document rollback procedures
- [ ] Document disaster recovery
- [ ] Create runbook for common incidents

**DEPLOYMENT.md outline:**
```markdown
# SIE-X Deployment Guide

## Prerequisites
- Kubernetes cluster (1.20+)
- kubectl configured
- Docker registry access
- Redis cluster

## Deployment Steps

### 1. Setup namespace
kubectl create namespace sie-x-prod

### 2. Configure secrets
kubectl create secret generic sie-x-secrets \
  --from-literal=jwt-secret=xxx \
  --from-literal=redis-password=yyy \
  -n sie-x-prod

### 3. Deploy
kubectl apply -k k8s/overlays/production

### 4. Verify
kubectl get pods -n sie-x-prod
kubectl logs -f deployment/sie-x-api -n sie-x-prod

### 5. Test
curl https://api.sie-x.com/health

## Scaling

### Manual scaling
kubectl scale deployment sie-x-api --replicas=10 -n sie-x-prod

### HPA already configured (3-20 replicas)

## Rollback

### To previous version
kubectl rollout undo deployment/sie-x-api -n sie-x-prod

### To specific revision
kubectl rollout history deployment/sie-x-api -n sie-x-prod
kubectl rollout undo deployment/sie-x-api --to-revision=3 -n sie-x-prod

## Disaster Recovery

### Database backup
# Redis automatic snapshots every 6 hours

### Restore from backup
kubectl exec -it redis-0 -n sie-x-prod -- redis-cli RESTORE ...

## Incident Response

### API is down
1. Check pod status: kubectl get pods
2. Check logs: kubectl logs
3. Check Redis: kubectl exec ... redis-cli PING
4. Rollback if recent deploy: kubectl rollout undo

### High latency
1. Check CPU/memory: kubectl top pods
2. Check HPA: kubectl get hpa
3. Manual scale if needed: kubectl scale
4. Check cache hit rate in Grafana
```

**Time:** 1 dag

---

**Phase 7 Total:** 3-4 dagar

---

## 📋 Phase 8: Packaging & Distribution (1-2 dagar)

### Prioritet: 🟢 MEDIUM

### 8.1 pip Package (PyPI)

**Tasks:**
- [ ] Create `setup.py` and `pyproject.toml`
- [ ] Define package metadata (name, version, description, author)
- [ ] Configure package dependencies
- [ ] Create `MANIFEST.in` for non-Python files
- [ ] Add CLI entry point (`siex` command)
- [ ] Build package: `python -m build`
- [ ] Test installation: `pip install dist/sie_x-*.whl`
- [ ] Upload to PyPI: `twine upload dist/*`

**setup.py:**
```python
# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.minimal.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="sie-x",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready semantic intelligence and keyword extraction platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robwestz/SEI-X",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "siex=sie_x.cli:main",
        ],
    },
    include_package_data=True,
)
```

**pyproject.toml:**
```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sie-x"
version = "1.0.0"
description = "Production-ready semantic intelligence and keyword extraction platform"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["nlp", "seo", "keywords", "semantic", "extraction"]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]

[project.urls]
Homepage = "https://github.com/robwestz/SEI-X"
Documentation = "https://sie-x.readthedocs.io"
Repository = "https://github.com/robwestz/SEI-X"
Changelog = "https://github.com/robwestz/SEI-X/blob/main/CHANGELOG.md"
```

**CLI:**
```python
# sie_x/cli.py
import click
from sie_x.core.simple_engine import SimpleSemanticEngine
from sie_x.core.multilang import MultiLangEngine
import json

@click.group()
def cli():
    """SIE-X - Semantic Intelligence Engine X"""
    pass

@cli.command()
@click.argument('text')
@click.option('--top-k', default=10, help='Number of keywords to extract')
@click.option('--lang', default=None, help='Language code (auto-detect if not specified)')
@click.option('--format', type=click.Choice(['json', 'text']), default='text')
def extract(text, top_k, lang, format):
    """Extract keywords from text"""
    if lang:
        engine = MultiLangEngine()
        keywords = engine.extract(text, language=lang, top_k=top_k)
    else:
        engine = SimpleSemanticEngine()
        keywords = engine.extract(text, top_k=top_k)

    if format == 'json':
        output = [kw.model_dump() for kw in keywords]
        click.echo(json.dumps(output, indent=2))
    else:
        for kw in keywords:
            click.echo(f"{kw.text:30} | score: {kw.score:.2f} | type: {kw.type}")

@cli.command()
def languages():
    """List supported languages"""
    engine = MultiLangEngine()
    for lang, model in engine.LANG_MODELS.items():
        click.echo(f"{lang:4} - {model}")

def main():
    cli()

if __name__ == '__main__':
    main()
```

**Usage after installation:**
```bash
# Install from PyPI
pip install sie-x

# Use CLI
siex extract "Machine learning is amazing" --top-k 5
siex extract "Hej världen" --lang sv --format json
siex languages
```

**Time:** 1 dag

---

### 8.2 Docker Hub Distribution

**Tasks:**
- [ ] Create automated Docker builds
- [ ] Push to Docker Hub
- [ ] Tag versions (latest, v1.0.0, stable)
- [ ] Add Docker Hub README

**Time:** 0.5 dag

---

### 8.3 Documentation Website

**Platform:** ReadTheDocs or MkDocs + GitHub Pages

**Tasks:**
- [ ] Setup MkDocs project
- [ ] Convert existing docs to markdown
- [ ] Add API reference (auto-generated from docstrings)
- [ ] Add tutorials and examples
- [ ] Deploy to GitHub Pages

**Time:** 0.5 dag

---

**Phase 8 Total:** 1-2 dagar

---

## 📋 Phase 9: Legal & Compliance (1 dag)

### Prioritet: 🟡 HIGH

**Requirements för kommersiell produkt:**

### 9.1 Legal Documents

**Tasks:**
- [ ] **Terms of Service (ToS)** - Define usage terms, liability, termination
- [ ] **Privacy Policy** - GDPR compliant data handling
- [ ] **Cookie Policy** - If web UI is used
- [ ] **API Terms** - Specific terms for API usage
- [ ] **SLA** (Service Level Agreement) - For enterprise customers

**Templates:**
- Use services like Termly.io or iubenda.com for templates
- Customize for SIE-X specific data handling
- Have lawyer review before launch

**Key clauses:**
- Data retention (90 days default)
- No warranty for free tier
- Rate limiting and fair use
- Prohibited uses (scraping, reselling)
- Termination policy

**Time:** 0.5 dag (template customization + review)

---

### 9.2 GDPR Compliance

**Tasks:**
- [ ] Data Processing Agreement (DPA) for EU customers
- [ ] Cookie consent banner (if applicable)
- [ ] Data breach notification procedure
- [ ] Appoint Data Protection Officer (DPO) if >250 employees
- [ ] Regular data audits

**Time:** 0.5 dag

---

**Phase 9 Total:** 1 dag

---

## 📋 Phase 10: Final Polish (1-2 dagar)

### Prioritet: 🟢 LOW

### 10.1 Documentation Review

**Tasks:**
- [ ] Review all README files for accuracy
- [ ] Update HANDOFF.md with v1.0 status
- [ ] Create CHANGELOG.md
- [ ] Add badges (build status, coverage, version)
- [ ] Proofread all documentation

---

### 10.2 Code Quality

**Tasks:**
- [ ] Run full linting pass (flake8, pylint, mypy)
- [ ] Format all code (black, isort)
- [ ] Remove dead code and TODOs
- [ ] Add missing docstrings
- [ ] Update type hints

---

### 10.3 Performance Baseline

**Tasks:**
- [ ] Run final load test
- [ ] Document performance metrics
- [ ] Create performance comparison chart (before vs. after optimizations)

---

**Phase 10 Total:** 1-2 dagar

---

## 📊 Summary & Timeline

| Phase | Priority | Time | Description |
|-------|----------|------|-------------|
| **Phase 3** | 🔴 CRITICAL | 3-4 dagar | Fix 5 known bugs |
| **Phase 4** | 🔴 CRITICAL | 5-7 dagar | Integration tests (>80% coverage) |
| **Phase 5** | 🟡 HIGH | 2-3 dagar | Security audit & auth |
| **Phase 6** | 🟢 MEDIUM | 2-3 dagar | Performance optimization |
| **Phase 7** | 🟡 HIGH | 3-4 dagar | K8s deployment + CI/CD |
| **Phase 8** | 🟢 MEDIUM | 1-2 dagar | PyPI package |
| **Phase 9** | 🟡 HIGH | 1 dag | Legal compliance |
| **Phase 10** | 🟢 LOW | 1-2 dagar | Final polish |
| **TOTAL** | | **19-28 dagar** | **~4 veckor** |

---

## 🎯 Recommended Execution Order

### Week 1: Critical Stability
1. Phase 3: Fix all 5 bugs (3-4 dagar)
2. Phase 4 (partial): Start integration tests (2-3 dagar)

**Deliverable:** Bug-free codebase, basic test coverage

---

### Week 2: Testing & Security
1. Phase 4 (complete): Finish integration tests (2-3 dagar)
2. Phase 5: Security audit (2-3 dagar)

**Deliverable:** 80%+ test coverage, secure API

---

### Week 3: Production Ready
1. Phase 7: Deployment setup (3-4 dagar)
2. Phase 6: Performance optimization (2-3 dagar)

**Deliverable:** Deployable to K8s, optimized performance

---

### Week 4: Polish & Launch
1. Phase 8: Package for PyPI (1-2 dagar)
2. Phase 9: Legal compliance (1 dag)
3. Phase 10: Final polish (1-2 dagar)

**Deliverable:** v1.0 ready for commercial launch

---

## ✅ Definition of Done (v1.0)

SIE-X v1.0 är **production-ready** när:

### Technical
- [x] Core funktionalitet fungerar (Phase 1-2.5)
- [ ] Alla 5 known bugs fixade
- [ ] >80% test coverage (integration tests)
- [ ] Security audit passed (no critical vulnerabilities)
- [ ] Load tested (1000 req/sec sustained)
- [ ] Deployable till Kubernetes
- [ ] CI/CD pipeline fungerar
- [ ] Monitoring och alerting konfigurerat

### Quality
- [ ] All documentation uppdaterad och korrekt
- [ ] Code quality checks passed (flake8, mypy, black)
- [ ] Performance benchmarks dokumenterade
- [ ] API documentation complete (Swagger/ReDoc)

### Business
- [ ] Terms of Service och Privacy Policy klara
- [ ] GDPR compliant
- [ ] pip-installable från PyPI
- [ ] Docker image på Docker Hub
- [ ] Pricing tiers definierade

### Operational
- [ ] Runbook för common incidents
- [ ] Backup och disaster recovery plan
- [ ] SLO/SLA definierade och measurable
- [ ] On-call rotation plan (för support)

---

## 🚀 Post-v1.0 Roadmap

**Efter första produktionsversionen är klar, fokusera på:**

### Phase 11: First 1000 Users (Month 2-3)
- Product Hunt launch
- Content marketing (blog posts)
- SEO for SIE-X website
- Collect user feedback
- Fix issues from production use

### Phase 12: Feature Expansion (Month 4-6)
- Build first 3 domain transformers:
  - LegalTransformer (law firms market)
  - FinancialTransformer (investment firms)
  - EcommerceTransformer (e-commerce stores)
- Add real-time SERP API integration
- Implement content generation API

### Phase 13: Enterprise Features (Month 7-12)
- White-label solution
- On-premise deployment
- Advanced analytics dashboard
- Team collaboration features
- Premium support tier

**Goal:** $500K-1M ARR within 12 months

---

## 📞 Questions & Decisions Needed

Before starting implementation, beslut krävs om:

1. **Deployment target:**
   - Self-hosted Kubernetes (on-prem eller cloud)?
   - Managed Kubernetes (GKE, EKS, AKS)?
   - Serverless (Google Cloud Run, AWS Lambda)?

2. **Database choice:**
   - PostgreSQL för user data?
   - MongoDB för flexible document storage?
   - Stick with Redis only?

3. **Pricing strategy:**
   - Freemium model (100 req/month free)?
   - Pay-as-you-go ($0.01/request)?
   - Subscription tiers (Pro $99, Business $299)?

4. **Legal:**
   - Company entity established?
   - Lawyer available for ToS/Privacy review?
   - GDPR DPO assigned?

5. **Team:**
   - Solo developer eller team?
   - Need DevOps support för K8s?
   - Marketing/sales resources available?

---

## 📝 Next Steps

**För att starta Phase 3 (bugfixar):**

```bash
# 1. Checkout på rätt branch (redan där)
git status

# 2. Skapa ny feature branch från current
git checkout -b feature/phase3-bugfixes

# 3. Börja med Issue #1 (language detection)
# Install fasttext
pip install fasttext

# Download language ID model
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# 4. Implementera fix i sie_x/core/multilang.py
# (Se detaljer i Phase 3.1 ovan)

# 5. Test
pytest sie_x/tests/test_multilang.py -v

# 6. Commit
git add .
git commit -m "fix: Upgrade language detection to fasttext (Issue #1)"

# 7. Push
git push -u origin feature/phase3-bugfixes
```

---

**Ready to begin? 🚀**

Vill du att jag börjar med Phase 3 (bugfixar) nu, eller har du frågor/ändringar till planen först?
