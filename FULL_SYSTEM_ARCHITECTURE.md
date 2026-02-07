# SIE-X: Full System Architecture Reference

**Document Version:** 2.1.0
**Senast uppdaterad:** 2026-02-07
**Scope:** Complete technical reference of the `sie_x` package — every file, every module, every known issue, and a forward-looking upgrade analysis.

This document serves as the **Master Reference** for AI agents and developers. It details every module, class, and capability within the SIE-X Engine, including advanced features often hidden in standard documentation.

---

## Complete File Index

Every file and folder in `sie_x/`, organized by module. Files marked with **[KEY]** are first-chain players that carry significant logic — described in full detail in their respective sections below.

```
sie_x/
├── __init__.py                          # Package root
├── FULL_SYSTEM_ARCHITECTURE.md          # ← This file
├── README.md                            # Swedish getting-started guide
├── SIEX_CAPABILITIES.md                 # AI agent capability guide
├── PACKAGE_CONTENTS.md                  # Complete file inventory
├── arcitechture.txt                     # Internal data flow notes (legacy)
├── project_builder.py                   # Zip-packaging script for project structure
├── project_packager.py                  # Collects all files into distributable zip
├── usecase.py                           # Runnable use case examples (legal, medical, finance)
│
├── core/                                # === LEVEL 0: CORE ENGINE ===
│   ├── __init__.py
│   ├── engine.py               [KEY]    # SemanticIntelligenceEngine — production engine (GPU, FAISS, async)
│   ├── simple_engine.py        [KEY]    # SimpleSemanticEngine — lightweight CPU-only engine
│   ├── models.py               [KEY]    # Pydantic data models (Keyword, Request, Response, etc.)
│   ├── extractors.py           [KEY]    # CandidateExtractor + TermFilter — candidate generation & noise removal
│   ├── resilience.py           [KEY]    # IntelligentRetry, circuit breaker, ResourceManager
│   ├── streaming.py            [KEY]    # StreamingExtractor — smart chunking + 3 merge strategies
│   ├── multilang.py                     # MultiLangEngine — 11-language support with auto-detection (FastText)
│   ├── utils.py                         # spaCy model auto-download helper
│   ├── test_core.py                     # Unit tests for core module
│   ├── interfaces.md                    # Interface contracts documentation
│   └── SEI_X_CORE_README.md            # Core module README
│
├── transformers/                        # === LEVEL 1: DOMAIN TRANSFORMERS ===
│   ├── __init__.py
│   ├── loader.py               [KEY]    # TransformerLoader — dynamic loading + hybrid multi-transformer systems
│   ├── seo_transformer.py      [KEY]    # SEOTransformer — bridge topics, intent alignment, content gaps
│   ├── medical_transformer.py  [KEY]    # MedicalTransformer — diagnosis, drug interactions, SOAP notes (~550 LOC)
│   ├── legal_transformer.py             # LegalTransformer — jurisdiction hierarchy, SFS/EU patterns (~140 LOC)
│   ├── financial_transformer.py         # FinancialTransformer — sentiment, trading signals, risk (~165 LOC)
│   └── creative_transformer.py          # CreativeTransformer — narrative analysis, character arcs (~200 LOC)
│
├── api/                                 # === LEVEL 2: REST API ===
│   ├── __init__.py
│   ├── minimal_server.py       [KEY]    # Main FastAPI app — 14 endpoints, auth, Prometheus, streaming
│   ├── server.py                        # Alternative server variant (production engine, async)
│   ├── routes.py                        # Extended routes — /analyze/url, /analyze/file, /keywords/search
│   ├── middleware.py            [KEY]    # AuthenticationMiddleware, RateLimitMiddleware, RequestTracingMiddleware
│   ├── auth.py                          # Basic JWT + bcrypt + RBAC (mock user DB)
│   └── README.md                        # API module documentation
│
├── auth/                                # === LEVEL 2: ENTERPRISE AUTH ===
│   ├── __init__.py
│   └── enterprise.py           [KEY]    # EnterpriseAuthManager — OIDC, SAML, LDAP + TokenManager
│
├── integrations/                        # === LEVEL 2: BUSINESS INTEGRATIONS ===
│   ├── __init__.py
│   └── bacowr_adapter.py       [KEY]    # BACOWRAdapter — intent alignment, smart constraints, compliance
│
├── sdk/                                 # === LEVEL 3: CLIENT SDK ===
│   ├── __init__.py
│   ├── README.md                        # SDK documentation
│   └── python/
│       ├── __init__.py
│       ├── client.py            [KEY]   # SIEXClient — async/sync HTTP client with retry
│       └── sie_x_sdk.py         [KEY]   # Enterprise SDK — WebSocket streaming, batch processor, OAuth
│
├── orchestration/                       # === LEVEL 3: LLM FRAMEWORK INTEGRATION ===
│   ├── __init__.py
│   └── langchain_integration.py [KEY]   # Embeddings, VectorStore, Retriever, Tools, QueryEngine
│
├── streaming/                           # === LEVEL 3: REAL-TIME STREAMING ===
│   ├── __init__.py
│   └── pipeline.py              [KEY]   # StreamingPipeline (Kafka) + WebSocketStreaming
│
├── multilingual/                        # === LEVEL 3: 100+ LANGUAGE SUPPORT ===
│   ├── __init__.py
│   └── engine.py                [KEY]   # MultilingualEngine — LaBSE + XLM-RoBERTa, auto-detect, language rules
│
├── chunking/                            # === LEVEL 0: DOCUMENT CHUNKING ===
│   ├── __init__.py
│   └── chunker.py               [KEY]   # DocumentChunker + SlidingWindowChunker — semantic boundary splitting
│
├── cache/                               # === LEVEL 0: CACHING ===
│   ├── __init__.py
│   ├── manager.py                       # CacheManager — in-memory LRU (OrderedDict, max_size)
│   └── redis_cache.py           [KEY]   # RedisCache, MemcachedCache, FallbackCache — distributed caching
│
├── graph/                               # === LEVEL 0: GRAPH (stub) ===
│   └── __init__.py                      # Package init (graph logic lives in core/engine.py)
│
├── monitoring/                          # === LEVEL 0: OBSERVABILITY ===
│   ├── __init__.py
│   ├── metrics.py                       # Prometheus counters, histogram, gauge + ASGI mount
│   └── observability.py         [KEY]   # ObservabilityManager — structlog + Prometheus + OpenTelemetry tracing
│
├── explainability/                      # === LEVEL 0: XAI ===
│   ├── __init__.py
│   └── xai.py                   [KEY]   # ExplainableExtractor — LIME, SHAP, counterfactuals, Plotly reports
│
├── audit/                               # === LEVEL 0: AUDIT & COMPLIANCE ===
│   ├── __init__.py
│   └── lineage.py                       # AuditManager — data lineage (NetworkX), GDPR/CCPA logging (SQLAlchemy)
│
├── export/                              # === LEVEL 0: DATA EXPORT ===
│   ├── __init__.py
│   └── formats.py                       # ExportManager — JSON, CSV (pandas), GraphML, raw embeddings
│
├── plugins/                             # === LEVEL 0: PLUGIN SYSTEM ===
│   ├── __init__.py
│   └── system.py                        # PluginManager — discover, load, execute hooks
│
├── agents/                              # === LEVEL 4: AUTONOMOUS AGENTS ===
│   ├── __init__.py
│   └── autonomous.py                    # Ray Serve agents — Monitor, Analyzer, Optimizer, Validator
│
├── automl/                              # === LEVEL 4: HYPERPARAMETER TUNING ===
│   ├── __init__.py
│   └── optimizer.py                     # AutoMLOptimizer — Optuna TPE, NAS prototype
│
├── testing/                             # === LEVEL 4: A/B TESTING ===
│   ├── __init__.py
│   └── ab_framework.py          [KEY]   # ABTestingFramework — Thompson sampling, statistical analysis (~450 LOC)
│
├── training/                            # === LEVEL 4: ACTIVE LEARNING ===
│   ├── __init__.py
│   └── active_learning.py       [KEY]   # ActiveLearningPipeline — contrastive fine-tuning, deployment gate
│
├── federated/                           # === LEVEL 4: FEDERATED LEARNING ===
│   ├── __init__.py
│   └── learning.py                      # FederatedLearningPipeline — PySyft, FedAvg aggregation
│
├── examples/                            # === CODE EXAMPLES ===
│   └── backlink_workflow.py             # End-to-end backlink automation demo
│
└── prompts/                             # === LLM PROMPTS ===
    └── writer_prompt.md                 # BACOWR system prompt for AI content writers
```

**Total: 22 directories, 38 Python source files, 9 documentation files**

---

## 1. Core Engine (`sie_x/core/`)

The heart of SIE-X. Two engines with different performance profiles, shared data models, and production-grade error handling.

### `engine.py` — Production Engine [KEY]

**Class:** `SemanticIntelligenceEngine`
**This is the main engine.** ~670 LOC of GPU-accelerated, async, production-ready extraction.

*   **Model Modes:**
    | Mode | Device | Features |
    |------|--------|----------|
    | `FAST` | CPU only | Basic NER + noun phrases |
    | `BALANCED` | CPU/GPU mixed | + FAISS vector search, clustering |
    | `ADVANCED` | Full GPU | + Subject-Verb-Object triplets, graph optimization |
    | `ULTRA` | Multi-GPU | All features, enterprise scale |

*   **Initialization Stack:**
    *   `SentenceTransformer` (default: `all-mpnet-base-v2`) on GPU if available
    *   `AutoTokenizer` for chunking
    *   spaCy (`en_core_web_lg` -> `en_core_web_md` -> `xx_ent_wiki_sm` fallback chain)
    *   FAISS index (GPU-accelerated `GpuIndexFlatL2` or CPU `IndexFlatL2`)
    *   `CacheManager`, `DocumentChunker`, `GraphOptimizer`, `MetricsCollector`

*   **Key Methods:**
    *   `extract_async(text, top_k, output_format, enable_clustering, min_confidence)` — Main entry point. Handles single text or batch `List[str]`. Outputs `object`, `string`, or `json`.
    *   `extract_multiple_advanced(texts, top_k_common, top_k_distinctive)` — Cross-document analysis: finds common keywords, distinctive keywords per doc, and document clusters.
    *   `finetune_domain(texts, gold_keywords, epochs)` — Domain adaptation (placeholder).

*   **Extraction Pipeline (10 steps):**
    1. Check MD5-based cache
    2. Chunk long documents (>5000 chars) via `DocumentChunker`
    3. Process chunks in parallel (`asyncio.gather`)
    4. Generate candidates: NER entities + noun phrases + SVO triplets (ADVANCED/ULTRA)
    5. Generate embeddings in batches with caching
    6. Build semantic graph (cosine similarity matrix, threshold 0.3)
    7. Graph optimization (ADVANCED/ULTRA modes)
    8. PageRank scoring (alpha=0.85)
    9. DBSCAN clustering (eps=0.3, cosine metric)
    10. Multi-factor final score: PageRank centrality + log frequency + entity type boost + cluster importance
    11. FAISS-based related term lookup
    12. Filter by confidence, cache results

### `simple_engine.py` — Lightweight Engine [KEY]

**Class:** `SimpleSemanticEngine` (~380 LOC)
*   CPU-only, no FAISS, no GPU, synchronous. Loads spaCy + SentenceTransformer (`all-MiniLM-L6-v2`).
*   **Pipeline:** Same conceptual flow (NER → embeddings → graph → PageRank) but without FAISS, DBSCAN, or async.
*   `extract(text, top_k, min_confidence, include_entities, include_concepts)` → `List[Keyword]`
*   Embedding-cache with MD5 keys.
*   Combined score: 70% PageRank + 30% frequency.
*   Alias: `SimpleExtractionEngine` (backwards compatibility).
*   **Use case:** Local development, testing, low-resource environments.

### `models.py` — Shared Data Models [KEY]

All Pydantic models used across API, SDK, and engine:

| Model | Purpose |
|-------|---------|
| `Keyword` | text, score, type, count, confidence, positions, embeddings, related_terms, semantic_cluster, metadata |
| `ExtractionOptions` | top_k, min_confidence, include_entities, include_concepts, language |
| `ExtractionRequest` | text (max 10,000 chars), optional URL, options, metadata |
| `ExtractionResponse` | keywords list, processing_time, version, metadata |
| `BatchExtractionRequest` | Up to 100 items with shared default options |
| `HealthResponse` | status (healthy/degraded/unhealthy), version, models_loaded, uptime |

All models include JSON Schema examples and field validation.

### `extractors.py` — Candidate Generation [KEY]

**Classes:** `CandidateExtractor`, `TermFilter` (~200 LOC)

*   `CandidateExtractor`:
    *   `extract_entities(doc)` — Named entities from spaCy, filtering numbers and short strings.
    *   `extract_noun_phrases(doc)` — Meaningful noun phrases, filtering stopwords/pronouns/determiners.
    *   `extract_key_phrases(text)` — Regex patterns: "X of Y", "X and Y", quoted phrases, parenthetical definitions.
    *   `normalize_text(text)` — Lowercase, collapse whitespace.

*   `TermFilter`:
    *   `is_valid(term)` — Checks length, regex stop patterns (URLs, hashtags, pure numbers), noise phrases ("click here", "read more", etc.), punctuation ratio.
    *   `filter_by_frequency(candidates, min_frequency, max_frequency)` — Frequency-based filtering.

*   **Standalone functions:** `merge_overlapping_phrases()`, `deduplicate_phrases()`.

### `resilience.py` — Error Handling & Resource Management [KEY]

**Classes:** `IntelligentRetry`, `ResourceManager` (~250 LOC)
**Decorator:** `@resilient_operation`

*   **Error Classification:** Maps exception types to severity levels:
    | Exception | Severity | Action |
    |-----------|----------|--------|
    | `ConnectionError` | TRANSIENT | Retry immediately |
    | `TimeoutError` | RECOVERABLE | Retry with backoff |
    | `MemoryError` | DEGRADED | Fall back to simpler method |
    | `ValueError` | CRITICAL | Fail fast |

*   `@resilient_operation(max_tries, backoff_factor, fallback)` — Combines:
    *   `circuitbreaker` (failure threshold + recovery timeout)
    *   `backoff.expo` (exponential backoff, max 5 min)
    *   Severity-based routing to fallback or fast-fail

*   `ResourceManager` — Async semaphore-based concurrency control + memory tracking with automatic cleanup.

### `streaming.py` — StreamingExtractor [KEY]

**Class:** `StreamingExtractor` (~640 LOC)

Chunked processing for large documents (>10K words):

*   **`ChunkConfig`** — chunk_size, overlap, min_chunk_size, strategy (`simple` or `smart`)
*   **Smart chunking** — Respects paragraph boundaries, Markdown headers, sentence structure. Falls back to word-based splitting for unstructured text. Handles oversized paragraphs by splitting on sentences.
*   **`extract_stream(text, top_k, min_confidence, merge_final)`** — Async generator yielding per-chunk results with progress percentage.
*   **3 merge strategies:**
    | Strategy | Behavior |
    |----------|----------|
    | `union` | All unique keywords, highest score wins |
    | `intersection` | Only keywords appearing in all chunks, averaged scores |
    | `weighted` | Frequency-boosted: `avg_score * (0.7 + 0.3 * chunk_frequency)` |
*   **Sync variant:** `extract_sync()` returns final merged result directly.

### Secondary Core Files

*   **`multilang.py`** — `MultiLangEngine`: 11 languages (en, sv, es, fr, de, it, pt, nl, el, nb, lt), FastText auto-detection with pattern matching fallback, on-demand model loading with LRU cache (max 5 models).
*   **`utils.py`** — `load_spacy_model(name)`: auto-downloads missing spaCy models via subprocess.
*   **`test_core.py`** — Unit tests for core extraction with mocks.
*   **`interfaces.md`** — Interface contracts between core and API/SDK/CLI components.
*   **`SEI_X_CORE_README.md`** — Core-specific README with architecture diagram.

---

## 2. Semantic Transformers (`sie_x/transformers/`)

Domain-specific intelligence layers that **inject** into the core engine via a shared pattern: each transformer wraps `engine.extract_async` with domain-enriched output while preserving original keywords.

### `loader.py` — Dynamic Transformer System [KEY]

**Class:** `TransformerLoader`

*   **Registry:** `{'seo': SEOTransformer, 'legal': LegalTransformer, 'medical': MedicalTransformer, 'financial': FinancialTransformer, 'creative': CreativeTransformer}`
*   `load_transformer(type, config)` — Instantiates and injects a transformer into the engine.
*   `create_hybrid_system(types)` — Loads multiple transformers simultaneously. Creates a hybrid `extract_async` that runs all transformers in parallel (`asyncio.gather`), then merges into combined result with `combined_insights` and `cross_domain_connections`.

> SEOTransformer is now registered as `'seo'` in the loader registry (fixed in U-05).

### `seo_transformer.py` [KEY]

**Class:** `SEOTransformer`
*   `find_bridge_topics(publisher, target)` — Cosine similarity between all entity pairs to find semantic pivot topics. Classifies as `Strong` (direct overlap), `Pivot` (thematic middleground), `Wrapper` (contextual framing). Returns Bridge Strength Score (0.0–1.0).
*   `analyze_target(...)` — Classifies search intent (Informational vs Commercial).
*   `find_content_gaps(...)` — Compares two texts to find missing semantic concepts.
*   **Injected methods:** `engine.find_bridge_topics`, `engine.analyze_target`, `engine.find_content_gaps`.

### `medical_transformer.py` [KEY]

**Class:** `MedicalTransformer` (~550 LOC) — Transforms SIE-X into **MedicalAI-X**

*   **Ontologies loaded:** ICD-11 (conditions), SNOMED-CT (symptoms), RxNorm (medications).
*   **Entity classification:** symptoms, conditions, medications, procedures, lab values, risk factors.
*   **Key capabilities:**
    *   `_differential_diagnosis()` — Bayesian reasoning on symptoms + patient history.
    *   `_check_drug_interactions()` — Detects known drug-drug interactions with severity levels.
    *   `_assess_severity()` — Critical/high/moderate/low based on keyword patterns.
    *   `_check_negation()` — Detects negated symptoms ("no fever", "denies pain").
    *   `_extract_temporal_info()` — Onset (sudden/gradual), duration, frequency.
    *   `_calculate_risk_scores()` — Cardiovascular risk, sepsis risk.
    *   `_check_red_flags()` — Chest pain, stroke, loss of consciousness, etc.
    *   `_generate_soap_note()` — Full SOAP clinical note generation.
*   **Injected methods:** `engine.diagnose`, `engine.check_drug_safety`, `engine.generate_soap_note`, `engine.calculate_risk_scores`.

### `legal_transformer.py`

**Class:** `LegalTransformer` (~140 LOC) — Transforms SIE-X into **LegalAI-X**

*   **Regex patterns:** Swedish law (SFS `\d{4}:\d+`), EU regulation, case citations (NJA, RH, HFD, AD), paragraph references.
*   **Jurisdiction hierarchy:** EU > Swedish Constitution > Swedish Law > Government Regulation > Agency Regulation.
*   **Key capabilities:** Legal entity classification, binding authority check, temporal validity, legal graph with conflict detection, legal weight calculation.
*   **Injected methods:** `engine.find_applicable_law`, `engine.check_legal_compliance`, `engine.generate_legal_memo`.

### `financial_transformer.py`

**Class:** `FinancialTransformer` (~165 LOC) — Transforms SIE-X into **FinanceAI-X**

*   **Patterns:** Tickers, currencies (USD/EUR/SEK), percentages, financial metrics (P/E, EPS, ROI, EBITDA).
*   **Key capabilities:** Company identification, entity sentiment analysis, financial event extraction with impact scoring, risk analysis, market impact prediction (beta-adjusted), trading signal generation (BUY/SELL with stop-loss/take-profit).
*   **Injected methods:** `engine.analyze_earnings_call`, `engine.detect_insider_trading`, `engine.generate_investment_thesis`, `engine.backtest_strategy`.

### `creative_transformer.py`

**Class:** `CreativeTransformer` (~200 LOC) — Transforms SIE-X into **CreativeAI-X**

*   **Narrative patterns:** Hero's Journey, Three-Act Structure, Kishōtenketsu.
*   **Key capabilities:** Character identification with archetype classification and arc tracing, theme extraction with symbolic representation, narrative structure analysis, plot twist suggestions (hidden identity, betrayal, thematic reversal), alternative narrative generation (perspective shifts, genre-bending), style analysis (voice, tone, rhythm, figurative language).
*   **Injected methods:** `engine.generate_story`, `engine.improve_dialogue`, `engine.create_character`, `engine.worldbuild`, `engine.plot_generator`.

---

## 3. REST API (`sie_x/api/`)

### `minimal_server.py` — Main API Server [KEY]

**Framework:** FastAPI (~875 LOC)
**Entry point:** `uvicorn sie_x.api.minimal_server:app`

This is the **primary production server**. Initializes `SimpleSemanticEngine`, `StreamingExtractor`, and `MultiLangEngine` on startup. Includes Prometheus middleware, rate limiting, CORS, and auth.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/extract` | POST | Extract keywords (auth required) |
| `/extract/batch` | POST | Batch extraction of multiple texts |
| `/extract/stream` | POST | SSE streaming (chunk-by-chunk results) |
| `/extract/multilang` | POST | Auto-detect language (11 languages) |
| `/token` | POST | JWT login (username/password → Bearer token) |
| `/health` | GET | Health check (engine status, uptime, loaded models) |
| `/models` | GET | List loaded models and cache size |
| `/stats` | GET | API usage statistics (total extractions, avg time, errors) |
| `/languages` | GET | Supported languages and detection stats |
| `/metrics` | GET | Prometheus metrics (ASGI mount) |
| `/knowledge/maps-routing-pack` | GET | Offline maps routing data (paginated, filterable) |
| `/knowledge/maps-routing-pack/page` | GET | Markdown page from maps vault (path-traversal protected) |
| `/` | GET | Root info with endpoint listing |
| `/docs` | GET | Auto-generated Swagger UI |

### `server.py` — Alternative Server

Alternative FastAPI server using the production `SemanticIntelligenceEngine` (GPU/async). Endpoints: `/extract`, `/extract/batch`, `/extract/stream` (NDJSON), `/analyze/multi`, `/health`. Background batch processing via `BackgroundTasks`.

### `routes.py` — Extended Routes

**Router prefix:** `/api/v1`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/analyze/url` | POST | Fetch URL (httpx), parse HTML (BeautifulSoup), extract keywords |
| `/api/v1/analyze/file` | POST | Upload file (.txt, .html, .md), extract keywords |
| `/api/v1/keywords/search` | GET | Search extracted keywords (Phase 1: mock data) |
| `/api/v1/stats` | GET | Detailed stats with success rate and requests/second |

*   `fetch_url_content()` — Async HTTP fetch with HTML-to-text conversion (script/style removal).
*   `extract_text_from_file()` — Supports UTF-8/Latin-1 text, HTML, Markdown.

### `middleware.py` — Production Middleware Stack [KEY]

Three middleware layers, all with graceful degradation:

1. **`AuthenticationMiddleware`** — JWT Bearer tokens + API Key authentication. Verifies against FallbackCache (Redis/Memcached). Skips auth for `/health`, `/metrics`, `/docs`.

2. **`RateLimitMiddleware`** — Dual rate limiting:
    *   Hourly limit (default 100/hour)
    *   Burst limit (default 10/minute)
    *   Adds `X-RateLimit-*` response headers.
    *   Skips gracefully if cache backend unavailable (dev mode).

3. **`RequestTracingMiddleware`** — Generates or propagates `X-Request-ID` (UUID4), logs start/completion with duration, structured logging via `structlog`.

### `auth.py` — Basic Authentication

JWT (python-jose, HS256) + bcrypt password hashing (passlib). Three roles: `ADMIN`, `USER`, `READ_ONLY`. Mock user database (2 users: admin/admin, user/user). Mock API keys (2 keys). FastAPI dependencies: `get_current_user()`, `get_current_active_user()`, `get_current_admin_user()`.

---

## 4. Enterprise Auth (`sie_x/auth/`)

### `enterprise.py` [KEY]

**Class:** `EnterpriseAuthManager` (~300 LOC)

Unified SSO authentication supporting three enterprise protocols:

| Protocol | Implementation | Dependencies |
|----------|---------------|-------------|
| **OIDC** | `authlib` OAuth2 client, supports Azure/Okta/Auth0 | `authlib` |
| **SAML** | `python3-saml` OneLogin integration | `python3-saml` |
| **LDAP** | `ldap3` bind + search, group-to-role mapping | `ldap3` |

**Class:** `TokenManager` — Redis-backed JWT creation, validation, and revocation with blacklisting. Configurable claims, refresh tokens.

**FastAPI dependencies:** `get_current_user()`, `require_roles(*roles)`, `require_groups(*groups)`.

---

## 5. Business Integrations (`sie_x/integrations/`)

### `bacowr_adapter.py` [KEY]

**Class:** `BACOWRAdapter`
*   `generate_smart_constraints(bridges, target_url)` — JSON rules for AI content writers: anchor text type, placement context, semantic keywords, must-mention entities, risk classification.
*   `intent_alignment` calculation: `min(bridge_strength + (topic_overlap * 0.3), 1.0)`.
*   Compliance checks: `link_density`, `anchor_risk`.
*   Used together with `prompts/writer_prompt.md` for BACOWR AI-skribent workflow.

---

## 6. Client SDK (`sie_x/sdk/python/`)

### `client.py` — Standard Client [KEY]

**Class:** `SIEXClient`
*   Async context manager (`async with SIEXClient(url) as client`).
*   `extract(text, top_k, min_confidence)` — Single extraction.
*   `extract_batch(texts)` — Batch extraction.
*   `analyze_url(url)`, `analyze_file(file_path)` — URL/file analysis.
*   `health()`, `stats()`, `models()`.
*   **Retry:** Exponential backoff (1s, 2s, 4s) with configurable `max_retries` (default 3).
*   **Sync wrappers:** `extract_sync()`, `batch_sync()`, `health_check_sync()`.
*   **Convenience function:** `extract_keywords(text, base_url)` — one-liner for quick usage.

### `sie_x_sdk.py` — Enterprise SDK [KEY]

**Class:** `SIEXClient` (extended version)
*   **Authentication:** `SIEXAuth` supporting API Key, JWT, and OAuth2 client credentials.
*   `extract_stream(text)` — WebSocket-based real-time streaming (yields chunk results).
*   `analyze_multiple(documents)` — Cross-document relationship analysis.
*   **Client-side caching:** SHA-256 hash keys, configurable `enable_caching`, max size.

**Class:** `SIEXBatchProcessor` — High-performance batch processor:
*   Semaphore-based concurrency control.
*   `process_files(directory, pattern)` — Glob-match and process all files.
*   Progress callback support.

---

## 7. LLM Orchestration (`sie_x/orchestration/`)

### `langchain_integration.py` [KEY]

Drop-in LangChain + LlamaIndex components (~375 LOC):

| Class | Replaces | Purpose |
|-------|----------|---------|
| `SIEXEmbeddings` | `OpenAIEmbeddings` | Embeddings via SIE-X engine |
| `SIEXTextSplitter` | `RecursiveCharacterTextSplitter` | Chunking via SIE-X |
| `SIEXVectorStore` | `FAISS`/`Chroma` | Vector search with keyword-boost scoring |
| `SIEXRetriever` | `VectorStoreRetriever` | Hybrid retrieval: vectors + keywords + clustering + reranking |
| `SIEXKeywordTool` | `BaseTool` | LangChain tool for keyword extraction |
| `SIEXMultiDocTool` | `BaseTool` | Multi-document analysis tool |
| `SIEXNodeParser` | LlamaIndex `NodeParser` | Node parsing with keyword metadata |
| `SIEXQueryEngine` | — | RAG engine: SIE-X retrieval + LLM answers |

`SIEXRetriever` re-ranks using keyword relevance + semantic cluster overlap. `SIEXVectorStore.similarity_search` scores: 30% keyword overlap + 70% semantic similarity.

---

## 8. Document Chunking (`sie_x/chunking/`)

### `chunker.py` [KEY]

**Class:** `DocumentChunker`
*   Token-based chunking using HuggingFace `PreTrainedTokenizer`.
*   Configurable `max_tokens` (default 512), `overlap_ratio` (default 0.1).
*   `respect_sentences=True` — Adjusts chunk boundaries to nearest sentence end.

**Class:** `SlidingWindowChunker` (extends DocumentChunker)
*   **Adaptive window sizing:** Calculates information density per token (inverse frequency as proxy).
*   Dense regions get smaller chunks (more detail), sparse regions get larger chunks.
*   `chunk_size = max_tokens * (0.7 + 0.3 * local_density)`.

---

## 9. Real-Time Streaming (`sie_x/streaming/`)

### `pipeline.py` [KEY]

**Class:** `StreamingPipeline` — Kafka-based real-time processing
*   **Stack:** `aiokafka` (consumer/producer) + `msgpack` (serialization) + `redis.asyncio` (caching).
*   4 parallel worker tasks consuming from input topic.
*   Micro-batching: processes batch when full (`batch_size`) or timeout (`batch_timeout`).
*   Redis-cached results (5 min TTL) to avoid reprocessing.
*   Dead letter queue for failed messages.
*   Prometheus counters for processed/failed messages.

**Class:** `WebSocketStreaming`
*   Handles WebSocket connections for real-time extraction.
*   Chunks text, extracts per-chunk, streams intermediate `chunk_result` events.
*   Sends `complete` event when done.

---

## 10. Caching (`sie_x/cache/`)

*   **`manager.py`** — `CacheManager`: In-memory LRU cache using `OrderedDict`. Simple get/set/clear with configurable `max_size`. 31 LOC. Used by the core engine for embedding and result caching.

*   **`redis_cache.py`** [KEY] — Distributed caching (~630 LOC):
    *   `CacheBackend` — Abstract base class defining the interface (get/set/delete/clear_all/get_stats/incr/expire).
    *   `RedisCache` — Primary distributed cache. Async via `redis.asyncio`, MD5-based key generation, configurable TTL (default 1h), key prefix `siex:`.
    *   `MemcachedCache` — Secondary option via `aiomcache`. Same interface, lighter weight.
    *   `FallbackCache` — Tries Redis → Memcached → graceful failure. Delete/clear operate on all backends. Supports `incr`/`expire` if active backend supports it.
    *   All operations return `None`/`False` on error instead of raising exceptions.

---

## 11. Monitoring & Observability (`sie_x/monitoring/`)

### `observability.py` — Full Observability Stack [KEY]

**Class:** `ObservabilityManager`

Three pillars integrated:

| Pillar | Technology | What it tracks |
|--------|-----------|----------------|
| **Logging** | `structlog` (JSON) | Every operation start/complete/error with callsite info |
| **Metrics** | `prometheus_client` | Counters (extractions, cache hits/misses), Histograms (duration by mode/size), Gauges (active ops, cache size), Info (model details) |
| **Tracing** | OpenTelemetry + OTLP exporter | Distributed traces to Jaeger/Tempo, spans for each operation |

*   `track_operation()` — Async context manager that automatically handles all three pillars.
*   `PerformanceMonitor` — Measures operation latency with success/failure tracking.

### `metrics.py` — Prometheus Metrics

6 pre-defined metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `sie_x_requests_total` | Counter | Requests per method/endpoint/status |
| `sie_x_request_duration_seconds` | Histogram | Latency per endpoint |
| `sie_x_errors_total` | Counter | Errors per type |
| `sie_x_active_requests` | Gauge | Currently processing |
| `sie_x_keywords_extracted_total` | Counter | Total keywords extracted |
| `sie_x_model_info` | Gauge | Loaded model information |

`get_metrics_app()` → ASGI app mounted at `/metrics`.
Legacy wrapper: `MetricsCollector` with timers.

---

## 12. Explainability (`sie_x/explainability/`)

### `xai.py` [KEY]

**Class:** `ExplainableExtractor` (~785 LOC)

`explain_extraction(text, keyword, detailed)` → `KeywordExplanation`:

| Component | Analysis Method |
|-----------|----------------|
| Linguistic | NER type, POS tags, term frequency |
| Semantic | Document coherence (cosine), semantic neighbors, cluster membership |
| Graph | Degree, betweenness, closeness, PageRank centrality |
| Context | Document position, co-occurring terms, syntactic role |
| LIME | Perturbation-based feature importance (100 samples) |
| Counterfactual | Impact of keyword removal, context sensitivity |

`KeywordExplanation` includes: score, confidence, decision path, alternative keywords.

**Class:** `ExplanationVisualizer` — Plotly visualizations:
*   `create_importance_chart()` — Horizontal bar chart of component importance
*   `create_evidence_sunburst()` — Hierarchical evidence drill-down
*   `create_decision_path_diagram()` — Step-by-step decision visualization
*   `create_explanation_report()` — Complete HTML report with all charts

---

## 13. Audit & Compliance (`sie_x/audit/`)

*   **`lineage.py`** — Data lineage + audit trail:
    *   `AuditLog` — SQLAlchemy model: event_type, user_id, session_id, resource_id, action, status, duration_ms, ip_address, user_agent. Dual index (user+time, resource).
    *   `DataLineage` — SQLAlchemy model for lineage nodes.
    *   `DataLineageNode` — Dataclass for in-memory lineage graph (NetworkX): input → process → output → model → cache.
    *   `AuditEventType` — 12 event types (extraction started/completed/failed, model loaded/updated, cache hit/miss, user action, config change, security event, data access/modification).

---

## 14. Export (`sie_x/export/`)

*   **`formats.py`** — `ExportManager`: 4 output formats:
    *   `to_json(keywords, metadata)` → Standard JSON
    *   `to_csv(keywords)` → pandas DataFrame → CSV
    *   `to_graphml(keywords, graph)` → NetworkX GraphML for Gephi/yEd
    *   `to_embeddings(keywords)` → Dict of raw vectors for Pinecone/Weaviate

---

## 15. Plugin System (`sie_x/plugins/`)

*   **`system.py`** — `PluginManager` (~440 LOC):
    *   **Interfaces:** `PluginInterface` (ABC), `ExtractorPlugin` (custom extraction), `ProcessorPlugin` (post-processing).
    *   **Discovery:** Scans `plugins/` directory, extracts YAML metadata from class docstrings.
    *   **Hooks:** `register_hook(name, callback)` / `execute_hook(name, *args)` — async-compatible event system.
    *   **Example plugins:** `DomainSpecificExtractor` (domain vocabulary boost), `AcademicCitationProcessor` (citation pattern detection and parsing).
    *   **Config schema:** JSON Schema for plugin configuration validation.

---

## 16. Autonomous Agents (`sie_x/agents/`)

*   **`autonomous.py`** — Built on **Ray Serve**:
    *   `BaseAgent` — ABC with async message queue, state dict, run loop.
    *   `AgentMessage` — sender, recipient, message_type, content, timestamp, priority.
    *   `MonitorAgent` — Watches system health metrics (latency, error rate, memory).
    *   `AnalyzerAgent` — Detects anomalies, finds root causes.
    *   `OptimizerAgent` — Executes automatic fixes (cache size, batch size tuning).
    *   `ValidatorAgent` — Runs A/B tests to verify that fixes improved performance.

---

## 17. AutoML (`sie_x/automl/`)

*   **`optimizer.py`** — `AutoMLOptimizer` (~200 LOC):
    *   **Optuna** with TPE sampler + MedianPruner.
    *   Tunes: mode, batch_size, max_chunk_size, embedding_model, graph_algorithm, graph_damping, semantic_threshold, clustering_eps, min_keyword_length.
    *   **NAS:** Neural Architecture Search prototype for model layer optimization.

---

## 18. A/B Testing (`sie_x/testing/`)

### `ab_framework.py` [KEY]

**Class:** `ABTestingFramework` (~450 LOC)

*   **Experiment lifecycle:** DRAFT → RUNNING → COMPLETED/STOPPED.
*   **Allocation strategies:**
    | Strategy | Method |
    |----------|--------|
    | RANDOM | Pure random allocation |
    | DETERMINISTIC | MD5 hash-based (consistent per user) |
    | ADAPTIVE | Thompson sampling (Beta-distribution multi-armed bandit) |

*   **Statistical analysis:** Welch's t-test, Cohen's d effect size, confidence intervals.
*   **Auto-stopping:** Ends when minimum sample size reached + statistical significance (p < 0.001).
*   `apply_winner()` — Automatically applies winning variant config to production engine.

**Class:** `ExperimentLibrary` — Pre-built experiments: embedding model comparison, chunking strategy test, threshold tuning.

---

## 19. Active Learning (`sie_x/training/`)

### `active_learning.py` [KEY]

**Class:** `ActiveLearningPipeline`

*   Collects `FeedbackSample` objects (text, extracted keywords, correct keywords, user rating).
*   **Retraining trigger:** Fires when feedback buffer reaches `retrain_threshold` (default 100).
*   **Training data:** Contrastive pairs — positive (correct keywords) + negative (incorrect extractions) + hard negatives.
*   **Fine-tuning:** `ContrastiveLoss` on SentenceTransformer, 3 epochs, warmup steps.
*   **Deployment gate:** New model must beat current by >2% F1 without losing >2% precision.
*   `get_uncertainty_samples()` — Identifies high-uncertainty candidates for human review (based on embedding variance).

---

## 20. Multilingual Engine (`sie_x/multilingual/`)

### `engine.py` [KEY]

**Class:** `MultilingualEngine` — Enterprise-grade multilingual support

*   **Language detection chain:** `langdetect` (fast) → `fasttext lid.176.bin` (accurate fallback).
*   **Base models:** LaBSE (109 languages) + XLM-RoBERTa for universal coverage.
*   **Per-language configs:** Specific spaCy models, sentence transformers, and tokenizers for en, es, zh, ar, and more.
*   **Language-specific post-processing:** Chinese word segmentation (jieba), Arabic RTL/diacritics normalization, Japanese kanji/kana handling.
*   Falls back to Stanza for languages without spaCy support.

---

## 21. Federated Learning (`sie_x/federated/`)

### `learning.py`

**Class:** `FederatedLearningPipeline` (~150 LOC) — Privacy-preserving distributed training

*   **Tech:** PySyft + PyTorch.
*   `register_client(id, data)` — Creates virtual worker, sends data, creates local model copy.
*   `train_round(epochs)` — Executes one federated round: local training on each client, then aggregate.
*   **Aggregation:** FedAvg (weighted average by sample count).
*   `evaluate_global_model(test_data)` — Accuracy, loss, sample count.

---

## 22. Root Utilities

| File | Purpose |
|------|---------|
| `project_builder.py` | Creates zip archive with complete SIE-X project structure |
| `project_packager.py` | Collects all Python files into distributable zip |
| `usecase.py` | Runnable use case examples: legal, medical, hybrid financial-legal |
| `arcitechture.txt` | Internal data flow notes (legacy text format) |

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│  Level 4: AUTONOMY                                              │
│  agents/autonomous.py · automl/optimizer.py                     │
│  testing/ab_framework.py · training/active_learning.py          │
│  federated/learning.py                                          │
├─────────────────────────────────────────────────────────────────┤
│  Level 3: ORCHESTRATION                                         │
│  orchestration/langchain_integration.py                         │
│  streaming/pipeline.py · sdk/python/ · multilingual/engine.py   │
├─────────────────────────────────────────────────────────────────┤
│  Level 2: INTEGRATIONS                                          │
│  integrations/bacowr_adapter.py                                 │
│  api/minimal_server.py · api/routes.py · api/middleware.py      │
│  auth/enterprise.py                                             │
├─────────────────────────────────────────────────────────────────┤
│  Level 1: TRANSFORMERS                                          │
│  transformers/seo · medical · legal · financial · creative      │
│  transformers/loader.py (hybrid systems)                        │
├─────────────────────────────────────────────────────────────────┤
│  Level 0: CORE ENGINE                                           │
│  core/engine.py · core/simple_engine.py · core/models.py       │
│  core/extractors.py · core/resilience.py · core/streaming.py   │
│  chunking/ · cache/ · monitoring/ · export/                     │
│  explainability/ · audit/ · plugins/ · graph/                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Known Issues & Technical Debt

Issues discovered during the 2.1.0 architecture review:

### Structural Issues

| # | Issue | Severity | Location |
|---|-------|----------|----------|
| 1 | **Two competing servers** — `minimal_server.py` (875 LOC, production-ready, 14 endpoints) vs `server.py` (simpler, uses production engine). Confusing which is "the" server. | Medium | `api/` |
| 2 | **Two competing multilingual systems** — `core/multilang.py` (11 languages, FastText) vs `multilingual/engine.py` (100+ languages, LaBSE). They don't reference each other. | Medium | `core/`, `multilingual/` |
| 3 | **Two competing auth systems** — `api/auth.py` (basic JWT + mock DB) vs `auth/enterprise.py` (OIDC/SAML/LDAP + Redis). No bridge between them. `minimal_server.py` imports from `api/auth.py`. | Medium | `api/`, `auth/` |
| 4 | ~~**SEOTransformer missing from loader registry**~~ | ~~Low~~ | **FIXED (U-05)** — Added to registry + proper imports |
| 5 | ~~**`graph/` package is a stub**~~ | ~~Low~~ | **FIXED (U-05)** — Proper docstring + future roadmap |
| 6 | ~~**`export/formats.py` has code artifacts**~~ | ~~Low~~ | **FIXED (U-05)** — Removed markdown block after line 65 |
| 7 | ~~**`plugins/system.py` references undefined `logger`**~~ | ~~Low~~ | **FIXED (U-05)** — Added `logging` + `asyncio` imports |

### Missing Infrastructure

| # | What's Missing | Impact |
|---|---------------|--------|
| 8 | **No real database** — Auth uses mock dicts, keyword search returns mock data, audit defines SQLAlchemy models but no session factory/migration. | High |
| 9 | **No configuration management** — Hardcoded defaults everywhere. No unified config, no `.env` support, no Pydantic Settings. | High |
| 10 | **No test suite** — Only `core/test_core.py` exists. No API tests, no transformer tests, no integration tests. | High |
| 11 | **No CLI** — `interfaces.md` mentions a CLI component but it was never built. | Medium |
| 12 | **No container/deployment story** — No Dockerfile, no docker-compose, no K8s manifests (referenced in `project_packager.py` but not in actual tree). | Medium |

---

## Major Upgrade Analysis

Based on a complete source-level review of all 38 Python files, these are the highest-impact upgrades SIE-X could receive, ranked by **value × feasibility**.

### Tier 1: Foundation (unblocks everything else)

#### U-01: Unified Configuration Layer
**Effort:** 1 day | **Impact:** Eliminates scattered hardcoded defaults

Every module has its own defaults: engine model names, cache TTLs, rate limits, auth secrets, Kafka brokers, Redis URLs, spaCy models. A `SIEXConfig` Pydantic Settings class with `.env` support would:

- Single source of truth for all configurable values
- Environment-based profiles (dev/staging/prod)
- Typed validation on startup, not at first use
- Makes testing trivial (inject test config)

```python
# New file: sie_x/config.py
class SIEXConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SIEX_", env_file=".env")

    # Engine
    engine_mode: str = "balanced"
    embedding_model: str = "all-mpnet-base-v2"
    spacy_model: str = "en_core_web_sm"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit_hourly: int = 100

    # Cache
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600

    # Auth
    secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
```

#### U-02: Real Database Layer
**Effort:** 2–3 days | **Impact:** Unlocks persistent state across entire system

Currently: mock dicts for auth, in-memory keyword store, audit models defined but no session. Needed:

- Async SQLAlchemy session factory (`create_async_engine`)
- Alembic migration setup
- Replace `FAKE_USERS_DB` with actual user table
- Make `/api/v1/keywords/search` work (currently returns mock)
- Connect `audit/lineage.py` to actual database
- SQLite for dev, PostgreSQL for prod (configurable via U-01)

#### U-03: Test Suite
**Effort:** 2–3 days | **Impact:** Enables safe refactoring and CI/CD

Currently only `core/test_core.py`. Needed:

- `pytest` + `pytest-asyncio` + `httpx` (TestClient)
- Unit tests for each transformer (mock engine, verify injected methods)
- API integration tests (TestClient against `minimal_server.py`)
- Engine tests with small fixture texts
- CI-ready (no GPU required, mock FAISS)

### Tier 2: Consolidation (remove confusion, reduce surface area)

#### U-04: Merge Duplicate Systems
**Effort:** 1–2 days | **Impact:** Eliminates the "which one do I use?" confusion

Three merges:

1. **Servers:** Keep `minimal_server.py` as the sole server (it's already the complete one). Move any production-engine-specific endpoints from `server.py` into it. Delete or archive `server.py`.

2. **Auth:** Make `api/auth.py` a thin wrapper that delegates to `auth/enterprise.py` when configured. In dev mode, use the mock DB. In prod, use OIDC/SAML/LDAP. One `get_current_user()` dependency that works in both modes.

3. **Multilingual:** Make `core/multilang.py` the "lite" implementation (11 languages, no heavy models) and `multilingual/engine.py` the "full" one. Add a factory function: `create_multilingual_engine(mode="lite"|"full")`.

#### U-05: Fix Known Code Issues
**Effort:** Half day | **Impact:** Code hygiene

- Add SEOTransformer to `loader.py` registry
- Fix undefined `logger` in `plugins/system.py`
- Clean artifact block from `export/formats.py`
- Either populate `graph/` or remove the package

### Tier 3: New Capabilities (expand what SIE-X can do)

#### U-06: LLM Integration Layer
**Effort:** 2–3 days | **Impact:** Transforms rule-based system into hybrid AI

Currently all "intelligence" is deterministic (NER, PageRank, regex patterns, Bayesian rules). Adding actual LLM calls would:

- **Intent classification:** Replace regex heuristics with Claude/GPT-based classification
- **Entity enrichment:** Ask LLM to resolve ambiguous entities, add context
- **Bridge topic reasoning:** Let LLM explain *why* a bridge exists (for BACOWR)
- **Transformer enhancement:** LLM-powered differential diagnosis, legal analysis, etc.
- **Convergence with synapse-engine:** This is exactly what synapse-engine Fas 1 needs

```python
# New file: sie_x/llm/client.py
class LLMClient:
    """Unified LLM interface for SIE-X."""

    def __init__(self, provider: str = "anthropic", model: str = "claude-sonnet-4-5-20250929"):
        ...

    async def classify_intent(self, text: str, candidates: List[str]) -> str: ...
    async def enrich_entities(self, entities: List[dict], context: str) -> List[dict]: ...
    async def explain_bridge(self, publisher: str, target: str, bridge: dict) -> str: ...
```

#### U-07: Persistent Vector Store
**Effort:** 1–2 days | **Impact:** Cross-session keyword memory + real RAG

Currently FAISS is in-memory (rebuilt every startup). Connecting to a persistent vector DB would:

- Enable keyword/embedding persistence across sessions
- Power the `/keywords/search` endpoint with real data
- Enable true RAG pipeline (not just in-memory `SIEXVectorStore`)
- Pinecone integration already conceptually fits via export → `to_embeddings()`

#### U-08: CLI Tool
**Effort:** 1 day | **Impact:** Developer experience

```bash
siex extract "Your text here" --top-k 10 --format json
siex serve --port 8000 --mode balanced
siex analyze-url https://example.com
siex tune --trials 50 --metric f1
siex export keywords.json --format csv
```

Typer or Click-based. Reads from U-01 config.

#### U-09: Container & CI/CD
**Effort:** 1 day | **Impact:** Deployability

- Multi-stage Dockerfile (builder + runtime)
- `docker-compose.yml` with SIE-X + Redis + Prometheus + Grafana
- GitHub Actions: lint → test → build → push
- Health check integration

### Tier 4: Future Vision

#### U-10: Real-time Learning Loop
**Effort:** 1 week+ | **Impact:** Self-improving system

Connect the pieces that already exist but aren't wired together:

```
User feedback → ActiveLearningPipeline → Fine-tuning →
  ABTestingFramework (validate) → Apply winner →
    AutoMLOptimizer (tune hyperparams) →
      MonitorAgent (watch metrics) → loop
```

All these components exist individually. The upgrade is the orchestration glue + a scheduler (could use Ray or Celery).

#### U-11: Multi-tenant SaaS Architecture
**Effort:** 1–2 weeks | **Impact:** Commercialization

- Tenant isolation (namespaced Redis keys, schema-per-tenant in DB)
- Usage metering and billing hooks
- Tenant-specific model fine-tuning (federated learning already exists)
- Admin dashboard API

---

## Upgrade Dependency Graph

```
U-01 (Config) ──────┬──→ U-02 (Database)
                     ├──→ U-03 (Tests)
                     ├──→ U-04 (Consolidation)
                     ├──→ U-06 (LLM Layer)
                     └──→ U-09 (Docker/CI)

U-02 (Database) ────┬──→ U-07 (Vector Store)
                     └──→ U-11 (Multi-tenant)

U-03 (Tests) ───────┬──→ U-04 (safe to refactor)
                     └──→ U-09 (CI pipeline)

U-06 (LLM Layer) ───┬──→ U-10 (Learning Loop)
                     └──→ synapse-engine Fas 1

U-04 + U-05 ────────→ Clean codebase for all Tier 3+
```

**Recommended start order:** U-01 → U-05 → U-03 → U-04 → U-02 → U-06

---

## Quick Reference for AI Agents

**"I need to extract keywords"** → `core/engine.py` (production) or `core/simple_engine.py` (lightweight)

**"I need SEO analysis"** → `transformers/seo_transformer.py` + `integrations/bacowr_adapter.py`

**"I need medical/legal/financial analysis"** → Load the appropriate transformer via `transformers/loader.py`

**"I need to call the API from Python"** → `sdk/python/client.py` (standard) or `sdk/python/sie_x_sdk.py` (enterprise)

**"I need to explain why a keyword was chosen"** → `explainability/xai.py`

**"I need real-time processing"** → `streaming/pipeline.py` (Kafka) or `api/minimal_server.py` `/extract/stream` (SSE)

**"I need to handle multiple languages"** → `multilingual/engine.py` (100+ languages) or `core/multilang.py` (11 languages)

**"I need to track data for GDPR"** → `audit/lineage.py`

**"I need to tune performance"** → `automl/optimizer.py` + `testing/ab_framework.py`

---

## Statistics

| Metric | Value |
|--------|-------|
| Directories | 22 |
| Python source files | 38 |
| Documentation files | 9 |
| Total LOC (estimated) | ~8,500 |
| Domain transformers | 5 |
| API endpoints | 14 |
| Prometheus metrics | 6 |
| Supported languages (core) | 11 |
| Supported languages (multilingual) | 100+ |
| Model modes | 4 (FAST, BALANCED, ADVANCED, ULTRA) |
| Known issues | 12 (4 fixed in U-05, 8 remaining) |
| Proposed upgrades | 11 (U-05 completed) |

**License:** Internal use / Proprietary.
