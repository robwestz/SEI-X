# SIE-X Package Contents

**Version:** 2.0.0
**Senast uppdaterad:** 2026-02-07
**Komplett teknisk referens:** [FULL_SYSTEM_ARCHITECTURE.md](FULL_SYSTEM_ARCHITECTURE.md)

Komplett inventering av varje fil och mapp i `sie_x/`-paketet. Filer markerade **[KEY]** är förstakedjesspelare med detaljerade beskrivningar; övriga filer har kortfattade beskrivningar av vad de gör.

---

## Filträd

```
sie_x/
├── __init__.py
├── FULL_SYSTEM_ARCHITECTURE.md      # Komplett teknisk arkitekturreferens
├── README.md                        # Projektöversikt, snabbstart, moduler
├── SIEX_CAPABILITIES.md             # AI-agent capability guide
├── PACKAGE_CONTENTS.md              # ← Denna fil
├── arcitechture.txt                 # Interna dataflödesnoteringar (legacy)
├── project_builder.py               # Zip-paketerings-script för projektstruktur
├── project_packager.py              # Samlar alla filer i ett distribuerbart zip-arkiv
├── usecase.py                       # Körbara use case-exempel (juridik, medicin, finans)
│
├── core/                            # [LEVEL 0] Kärnmotorer
│   ├── __init__.py
│   ├── engine.py                    # [KEY] Produktionsmotor — GPU, FAISS, async
│   ├── simple_engine.py             # [KEY] Lättviktsmotor — CPU, synkron
│   ├── models.py                    # [KEY] Pydantic-modeller — hela systemets datakontrakt
│   ├── extractors.py                # [KEY] Kandidatgenerering + brusfiltrering
│   ├── resilience.py                # [KEY] Circuit breaker, retry, resurshantering
│   ├── streaming.py                 # [KEY] StreamingExtractor — chunked processing
│   ├── multilang.py                 # 11 språk med auto-detektering (FastText)
│   ├── utils.py                     # spaCy-modell auto-nedladdning
│   ├── test_core.py                 # Standalone-tester med mocks
│   ├── interfaces.md                # Interfacekontrakt mellan core och övriga moduler
│   └── SEI_X_CORE_README.md         # Core-specifik README
│
├── transformers/                    # [LEVEL 1] 5 domäntransformers
│   ├── __init__.py
│   ├── loader.py                    # [KEY] TransformerLoader — dynamisk laddning + hybrid
│   ├── seo_transformer.py           # [KEY] SEO — bryggor, intent, content gaps
│   ├── medical_transformer.py       # [KEY] Medicin — differentialdiagnos, SOAP, ICD-11
│   ├── legal_transformer.py         # Juridik — SFS/EU-mönster, hierarki, konflikter
│   ├── financial_transformer.py     # Finans — sentiment, trading signals, risk
│   └── creative_transformer.py      # Kreativt — narrativstruktur, karaktärer, plot
│
├── api/                             # [LEVEL 2] REST API
│   ├── __init__.py
│   ├── minimal_server.py            # [KEY] Huvud-FastAPI-app (alla endpoints)
│   ├── server.py                    # Alternativ server (async engine-baserad)
│   ├── routes.py                    # Utökade routes: URL-analys, filuppladdning, sökning
│   ├── auth.py                      # JWT/API-key autentisering, roller, mock-DB
│   ├── middleware.py                # [KEY] Auth + rate limiting + request tracing
│   └── README.md                    # API-modulens dokumentation
│
├── auth/                            # [LEVEL 2] Enterprise SSO
│   ├── __init__.py
│   └── enterprise.py               # [KEY] OIDC + SAML + LDAP, TokenManager (Redis)
│
├── integrations/                    # [LEVEL 2] Externa kopplingar
│   ├── __init__.py
│   └── bacowr_adapter.py           # [KEY] Backlink-automation: intent alignment, constraints
│
├── sdk/                             # [LEVEL 3] Python SDK:er
│   ├── __init__.py
│   ├── README.md                    # SDK-dokumentation
│   └── python/
│       ├── __init__.py
│       ├── client.py                # [KEY] Standard-SDK — async/sync, retry, batch
│       └── sie_x_sdk.py            # [KEY] Enterprise-SDK — WebSocket, OAuth, caching
│
├── chunking/                        # [LEVEL 0] Intelligent dokumentsplittning
│   ├── __init__.py
│   └── chunker.py                  # [KEY] DocumentChunker + SlidingWindowChunker
│
├── cache/                           # [LEVEL 0] Cachning
│   ├── __init__.py
│   ├── manager.py                   # In-memory LRU-cache (OrderedDict)
│   └── redis_cache.py              # [KEY] Redis + Memcached + FallbackCache
│
├── graph/                           # [LEVEL 0] Semantisk graf
│   └── __init__.py                  # Package init (graflogik inbyggd i engine.py)
│
├── monitoring/                      # [LEVEL 0] Observability
│   ├── __init__.py
│   ├── metrics.py                   # Prometheus-counters, histogram, gauge, ASGI-mount
│   └── observability.py            # [KEY] structlog + Prometheus + OpenTelemetry tracing
│
├── explainability/                  # [LEVEL 0] Förklarbar AI (XAI)
│   ├── __init__.py
│   └── xai.py                      # [KEY] LIME, SHAP, counterfactuals, Plotly-visualiseringar
│
├── audit/                           # [LEVEL 0] Spårbarhet & GDPR
│   ├── __init__.py
│   └── lineage.py                  # Data lineage (NetworkX-graf), audit log (SQLAlchemy)
│
├── export/                          # [LEVEL 0] Dataexport
│   ├── __init__.py
│   └── formats.py                   # JSON, CSV (pandas), GraphML, raw embeddings
│
├── plugins/                         # [LEVEL 0] Pluginsystem
│   ├── __init__.py
│   └── system.py                   # PluginManager, ExtractorPlugin, ProcessorPlugin, hooks
│
├── orchestration/                   # [LEVEL 3] LangChain/LlamaIndex
│   ├── __init__.py
│   └── langchain_integration.py    # [KEY] Embeddings, VectorStore, Retriever, Tools
│
├── streaming/                       # [LEVEL 3] Real-time processing
│   ├── __init__.py
│   └── pipeline.py                 # [KEY] Kafka-pipeline + WebSocket streaming
│
├── multilingual/                    # [LEVEL 3] 100+ språk
│   ├── __init__.py
│   └── engine.py                   # [KEY] LaBSE + XLM-RoBERTa, auto-detect, per-språk config
│
├── agents/                          # [LEVEL 4] Autonoma agenter
│   ├── __init__.py
│   └── autonomous.py               # Ray Serve: Monitor, Analyzer, Optimizer, Validator
│
├── automl/                          # [LEVEL 4] Hyperparameter-optimering
│   ├── __init__.py
│   └── optimizer.py                # Optuna TPE + MedianPruner, NAS-prototyp
│
├── testing/                         # [LEVEL 4] Experiment
│   ├── __init__.py
│   └── ab_framework.py             # [KEY] Thompson sampling, Welch's t-test, auto-deploy
│
├── training/                        # [LEVEL 4] Aktiv inlärning
│   ├── __init__.py
│   └── active_learning.py          # [KEY] Contrastive fine-tuning, deployment gate, uncertainty
│
├── federated/                       # [LEVEL 4] Privacy-preserving ML
│   ├── __init__.py
│   └── learning.py                  # PySyft + FedAvg aggregering
│
├── examples/                        # Körbara exempel
│   └── backlink_workflow.py         # Komplett SEO backlink-flöde med SDK + BACOWR
│
└── prompts/                         # Promptmallar
    └── writer_prompt.md             # BACOWR AI-skribent systemprompt
```

---

## Level 0: Core Engine

### `core/engine.py` [KEY] — Produktionsmotor (~670 LOC)

`SemanticIntelligenceEngine` — Full GPU-accelererad motor med 10-stegs pipeline:

1. **NER + noun phrases** (spaCy) — kandidatgenerering
2. **Embeddings** (SentenceTransformer) — semantisk vektorisering
3. **Cosine similarity matrix** — alla par
4. **Graph** (NetworkX) — semantiskt nätverk
5. **PageRank** (alpha=0.85) — rankning
6. **DBSCAN clustering** — gruppering
7. **FAISS vektor-index** — snabb sökning (GPU om tillgängligt)
8. **Multi-factor scoring** — kombinerad poäng
9. **MMR deduplication** — maximal marginal relevans
10. **Output formatting** — `json` / `object` / `string`

4 modlägen: `FAST` (CPU), `BALANCED` (mixed), `ADVANCED` (full GPU + SVO-tripletter), `ULTRA` (multi-GPU).

Asynkron: `extract_async()`, `extract_multiple_advanced()`. Inbyggd embedding-cache.

### `core/simple_engine.py` [KEY] — Lättviktsmotor (~380 LOC)

`SimpleSemanticEngine` — CPU-only, synkron. Samma pipeline (NER → embeddings → graf → PageRank) men utan FAISS, GPU, DBSCAN eller async. Perfekt för utveckling och tester.

- `extract(text, top_k, min_confidence)` → `List[Keyword]`
- Embedding-cache med MD5-nycklar
- Kombinerad score: 70% PageRank + 30% frekvens
- Alias: `SimpleExtractionEngine` (bakåtkompatibilitet)

### `core/models.py` [KEY] — Datamodeller

Pydantic-modeller som hela systemet bygger på:

| Modell | Syfte |
|--------|-------|
| `Keyword` | Nyckelord med score, type, count, confidence, positions, embeddings, related_terms, semantic_cluster |
| `ExtractionOptions` | top_k, min_confidence, include_entities, include_concepts |
| `ExtractionRequest` | text + options + url + metadata |
| `ExtractionResponse` | keywords + processing_time + version + metadata |
| `BatchExtractionRequest` | Lista av ExtractionRequest + gemensamma options |
| `HealthResponse` | status, version, models_loaded, uptime |

### `core/extractors.py` [KEY] — Kandidatgenerering (~200 LOC)

Två klasser som hanterar steg 1 i pipelinen:

- **`CandidateExtractor`** — Genererar kandidater via tre metoder: Named Entities (spaCy NER), Noun Phrases (spaCy chunks), Regex Key Phrases (mönstermatchning).
- **`TermFilter`** — Rensar brus: stoppord, för korta termer, frekvensfiltrering, överlappande fraser.
- **`merge_overlapping_phrases()`** och **`deduplicate_phrases()`** — Utility-funktioner.

### `core/resilience.py` [KEY] — Produktionsresiliens (~250 LOC)

Tre viktiga komponenter:

- **`IntelligentRetry`** — Klassificerar fel i 4 nivåer: `TRANSIENT` (nätverksfel, retry), `RECOVERABLE` (begränsad retry), `DEGRADED` (graceful fallback), `CRITICAL` (ingen retry). Exponentiell backoff med jitter.
- **`@resilient_operation`** — Decorator som kombinerar circuit breaker + retry + timeout. Wrappas runt valfri funktion.
- **`ResourceManager`** — Async semaphore för concurrency-kontroll + minnesövervakning.

### `core/streaming.py` [KEY] — StreamingExtractor (~640 LOC)

Chunked processing för stora dokument (>10K ord):

- **`ChunkConfig`** — chunk_size, overlap, min_chunk_size, strategy (`simple` eller `smart`)
- **`StreamingExtractor`** — Asynkron generator (`extract_stream()`) som yieldar chunk-resultat med progress-procent. Tre merge-strategier: `union` (alla), `intersection` (gemensamma), `weighted` (frekvensbaserad).
- **Smart chunking** — Respekterar styckesgränser, Markdown-headers, meningsstruktur. Fallback till ordbaserad splittning för oregelbunden text.
- **Synkron variant** — `extract_sync()` returnerar slutresultat direkt utan streaming.

### `core/multilang.py` — Flerspråkig grundmotor

`MultiLangEngine` — 11 språk med FastText auto-detektering. Laddar spaCy-modeller on demand med LRU-cache (max 5 modeller i minnet). Stödda språk: en, sv, es, fr, de, it, pt, nl, el, nb, lt.

### Övriga core-filer

| Fil | Syfte |
|-----|-------|
| `utils.py` | `load_spacy_model()` med automatisk nedladdning om modellen saknas |
| `test_core.py` | Standalone-tester med mocks — validerar modeller och engine utan GPU |
| `interfaces.md` | Definierar interfacekontrakt mellan core och API/SDK/CLI |
| `SEI_X_CORE_README.md` | Core-specifik README med arkitekturdiagram och snabbstart |

---

## Level 1: Domäntransformers

Alla transformers följer **inject-mönstret**: de kopplas till en engine via `TransformerLoader` och injicerar domänspecifika metoder och logik direkt på engine-objektet.

### `transformers/loader.py` [KEY] — TransformerLoader

Dynamisk laddning och hybrid-system:

- `load_transformer('medical')` — Laddar en transformer, injicerar dess metoder
- `create_hybrid_system(['legal', 'financial'])` — Laddar flera, skapar en kombinerad `extract`-funktion som kör alla transformers och mergar resultat med `combined_insights` och `cross_domain_connections`

### `transformers/seo_transformer.py` [KEY] — SEOTransformer

Kärnan i SEO-användningen av SIE-X:

- **`find_bridge_topics(pub_data, tgt_data)`** — Hittar semantiska bryggor mellan publisher och target via cosine similarity. Klassificerar som `Strong` (direkt överlapp), `Pivot` (tematisk mellanlandning), `Wrapper` (kontextuell inramning). Returnerar Bridge Strength Score (0.0–1.0).
- **Intent Alignment** — Formel: `min(bridge_strength + (topic_overlap * 0.3), 1.0)`. Styr ankartextaggressivitet.
- **Content gaps** — Identifierar ämnen som saknas hos publishern men finns hos target.

### `transformers/medical_transformer.py` [KEY] — MedicalTransformer (~550 LOC)

Fullständig klinisk AI-pipeline:

- **Ontologier:** ICD-11, SNOMED-CT, RxNorm
- **Differentialdiagnos:** Bayesiansk reasoning — beräknar sjukdomssannolikhet givet symptom + patienthistorik
- **Läkemedelsinteraktioner:** Detekterar kontraindikationer med allvarlighetsgrad
- **Negationsdetektering:** Förstår "denies pain", "no fever"
- **Red flags:** Bröstsmärta, stroke-symtom, sepsis-tecken
- **SOAP-anteckning:** Genererar Subjective/Objective/Assessment/Plan
- **Risk scores:** Kardiovaskulär risk, sepsis risk

Injicerade metoder: `engine.diagnose()`, `engine.check_drug_safety()`, `engine.generate_soap_note()`, `engine.calculate_risk_scores()`

### Övriga transformers

| Transformer | LOC | Nyckelförmåga |
|-------------|-----|---------------|
| `legal_transformer.py` | ~140 | SFS/EU-mönster, jurisdiktionshierarki (EU > Grundlag > Lag > Förordning > Föreskrift), juridisk graf, konfliktdetektering. Injicerar: `find_applicable_law()`, `check_legal_compliance()`, `generate_legal_memo()` |
| `financial_transformer.py` | ~165 | Ticker/valuta/metric-mönster, sentiment, marknadspåverkan, trading-signaler. Injicerar: `analyze_earnings_call()`, `detect_insider_trading()`, `generate_investment_thesis()`, `backtest_strategy()` |
| `creative_transformer.py` | ~200 | Narrativmönster (Hero's Journey, Three-Act, Kishōtenketsu), karaktärsanalys, plot twists, alternativa narrativ. Injicerar: `generate_story()`, `improve_dialogue()`, `create_character()`, `worldbuild()`, `plot_generator()` |

---

## Level 2: API, Auth & Integrations

### `api/minimal_server.py` [KEY] — FastAPI Huvud-app (~875 LOC)

Produktionsredo API-server. Endpoints:

| Endpoint | Metod | Beskrivning |
|----------|-------|-------------|
| `/extract` | POST | Extrahera nyckelord (autentisering krävs) |
| `/extract/batch` | POST | Batchbearbetning av flera texter |
| `/extract/stream` | POST | SSE streaming chunk för chunk |
| `/extract/multilang` | POST | Auto-detektering av språk (11 språk) |
| `/token` | POST | JWT-inloggning (username/password) |
| `/health` | GET | Hälsokontroll |
| `/models` | GET | Laddade modeller och cachestatus |
| `/stats` | GET | API-statistik |
| `/languages` | GET | Stödda språk och detekteringsstatistik |
| `/metrics` | GET | Prometheus-metrics (ASGI mount) |
| `/knowledge/maps-routing-pack` | GET | Offline maps routing-data (paginerad) |
| `/knowledge/maps-routing-pack/page` | GET | Markdown-sida från maps vault |

Startar vid uppstart: `SimpleSemanticEngine`, `StreamingExtractor`, `MultiLangEngine`.
In-memory rate limiting (10 req/s per IP). CORS aktiverat. Prometheus-middleware.

### `api/middleware.py` [KEY] — Tre middleware-klasser

| Middleware | Syfte |
|------------|-------|
| `AuthenticationMiddleware` | JWT Bearer + API Key (via FallbackCache). Skippar `/health` och `/docs`. |
| `RateLimitMiddleware` | 100 requests/timme + 10/minut burst. Graceful skip vid cache-fel. FallbackCache som backend. |
| `RequestTracingMiddleware` | Genererar `X-Request-ID` (UUID4), injicerar i structlog-kontext. |

### `api/auth.py` — Grundläggande autentisering

JWT (python-jose) + bcrypt (passlib). Tre roller: `ADMIN`, `USER`, `READ_ONLY`. Mock-användardatabas (ersätts med riktig DB i produktion). Dependencies: `get_current_user()`, `get_current_active_user()`, `get_current_admin_user()`.

### `api/routes.py` — Utökade routes (`/api/v1/`)

| Endpoint | Syfte |
|----------|-------|
| `POST /api/v1/analyze/url` | Hämtar URL (httpx), parsar HTML (BeautifulSoup), extraherar nyckelord |
| `POST /api/v1/analyze/file` | Laddar upp fil (.txt/.html/.md), extraherar nyckelord |
| `GET /api/v1/keywords/search` | Sök bland extraherade nyckelord (mock i Phase 1) |
| `GET /api/v1/stats` | Detaljerad statistik med success rate och req/s |

### `auth/enterprise.py` [KEY] — Enterprise SSO (~300 LOC)

`EnterpriseAuthManager` — Stödjer tre SSO-protokoll:

| Protokoll | Bibliotek | Providers |
|-----------|-----------|-----------|
| OIDC | authlib | Azure AD, Okta, Auth0 |
| SAML | python3-saml | OneLogin, generisk |
| LDAP | ldap3 | Active Directory, OpenLDAP |

`TokenManager` — Redis-backed JWT med refresh tokens, token-revokering (blacklist), konfigurerbara claims.

FastAPI dependencies: `get_current_user()`, `require_roles('admin')`, `require_groups('engineering')`.

### `integrations/bacowr_adapter.py` [KEY] — Backlink-automation

`BACOWRAdapter` — Kopplar SIE-X bridge-analys till praktisk backlink-strategi:

- **`generate_smart_constraints(bridges, target_url)`** — Genererar regler för AI-skribent: ankartexttyp, placeringskontext, semantiska nyckelord, must-mention entities, riskklassificering.
- **Intent Alignment** — Matematisk beräkning: `min(bridge_strength + (topic_overlap * 0.3), 1.0)`. Styr hur aggressiv ankartexten får vara.
- Används med `prompts/writer_prompt.md` för att instruera en LLM-skribent.

---

## Level 3: SDK, Orchestration, Streaming, Multilingual

### `sdk/python/client.py` [KEY] — Standard Python SDK

`SIEXClient` — Async-first HTTP-klient (httpx):

- `extract(text, top_k, min_confidence)` — Enskild extraktion
- `extract_batch(texts)` — Batch
- `analyze_url(url)` / `analyze_file(path)` — URL/fil-analys
- `health()` / `stats()` — Monitoring
- Exponentiell backoff-retry (3 försök)
- Synkrona wrappers: `extract_sync()`, `batch_sync()`
- Convenience-funktion: `extract_keywords("text", api_url=...)`

### `sdk/python/sie_x_sdk.py` [KEY] — Enterprise SDK

`SIEXClient` (enterprise) — Utökar standard-SDK med:

- **`SIEXAuth`** — 3 auth-metoder: API Key, JWT, OAuth2 (client credentials)
- **WebSocket streaming** — `extract_stream()` yield:ar chunk-resultat i realtid
- **Client-side caching** — SHA-256 hash-baserad, konfigurerbar max size
- **`SIEXBatchProcessor`** — Parallell batch-bearbetning med semaphore-kontrollerad concurrency

### `orchestration/langchain_integration.py` [KEY] — LangChain + LlamaIndex

Drop-in replacements för standard LangChain-komponenter:

| Klass | Ersätter | Syfte |
|-------|----------|-------|
| `SIEXEmbeddings` | `OpenAIEmbeddings` | Embeddings via SIE-X engine |
| `SIEXTextSplitter` | `RecursiveCharacterTextSplitter` | Chunking via SIE-X |
| `SIEXVectorStore` | `FAISS`/`Chroma` | Vektorsökning med nyckelords-boost |
| `SIEXRetriever` | `VectorStoreRetriever` | Hybrid retrieval: vektorer + nyckelord + klustring + reranking |
| `SIEXKeywordTool` | `BaseTool` | LangChain-tool för nyckelordsextraktion |
| `SIEXMultiDocTool` | `BaseTool` | Multi-dokument-analys |
| `SIEXNodeParser` | LlamaIndex `NodeParser` | Nod-parsning med nyckelord-metadata |
| `SIEXQueryEngine` | — | RAG-engine: SIE-X retrieval + LLM svar |

### `streaming/pipeline.py` [KEY] — Real-time processing

Två streaming-varianter:

- **`StreamingPipeline`** (Kafka) — aiokafka + msgpack + redis.asyncio. 4 parallella workers, micro-batching, Redis-cache per meddelande, dead letter queue (DLQ) vid fel, Prometheus-metrics.
- **`WebSocketStreaming`** — Chunk-för-chunk WebSocket-resultat i realtid.

### `multilingual/engine.py` [KEY] — 100+ språk

`MultilingualEngine` — Enterprise-klass flerspråkig motor:

- **Modeller:** LaBSE (109 språk) + XLM-RoBERTa base
- **Språkdetektering:** Kedjad: langdetect → fasttext fallback
- **Per-språk konfiguration:** Anpassade stop words, embedding-parametrar, tokenizer-val
- **Språkspecifik post-processing:**
  - Kinesiska: jieba-segmentering
  - Arabiska: RTL-normalisering
  - Japanska: Kanji/kana-separation

---

## Level 4: Autonomy

### `agents/autonomous.py` — Ray Serve Agenter

Fyra samverkande agenter (multi-agent loop):

| Agent | Roll |
|-------|------|
| `MonitorAgent` | Övervakar latens, felfrekvens, minnesanvändning |
| `AnalyzerAgent` | Hittar grundorsaker till prestandaproblem |
| `OptimizerAgent` | Justerar konfiguration i realtid (cache-storlek, batch size) |
| `ValidatorAgent` | Kör A/B-tester för att validera optimeringar |

Meddelande-system: `AgentMessage` med prioritet. `BaseAgent` med async message queue + task loop.

### `testing/ab_framework.py` [KEY] — A/B-testning (~450 LOC)

`ABTestingFramework` — Fullständig experiment-infrastruktur:

- **Livscykel:** DRAFT → RUNNING → COMPLETED/STOPPED
- **3 allokeringsstrategier:** RANDOM, DETERMINISTIC (hash-baserad), ADAPTIVE (Thompson sampling med Beta-distribution)
- **Statistik:** Welch's t-test, Cohen's d effect size, konfidensintervall
- **Auto-stopping:** Stoppar tidigt om en variant visar sig överlägsen
- **`apply_winner()`** — Applicerar vinnande konfiguration direkt på produktionsmotorn
- **`ExperimentLibrary`** — Fördefinierade experiment (model comparison, threshold tuning, etc.)

### `training/active_learning.py` [KEY] — Aktiv inlärning

`ActiveLearningPipeline` — Kontinuerlig förbättring från användarfeedback:

- **Feedback-buffer** — Samlar korrekta/felaktiga extraktioner
- **Trigger:** Startar automatiskt efter N samples (default 100)
- **Contrastive fine-tuning:** Tränar på positiva/negativa/hard negative-par med `ContrastiveLoss`
- **Deployment gate:** Ny modell måste slå nuvarande med >2% F1 utan att tappa >2% precision
- **`get_uncertainty_samples()`** — Identifierar samples med hög osäkerhet för mänsklig granskning

### Övriga Level 4-moduler

| Modul | Fil | Syfte |
|-------|-----|-------|
| `automl/optimizer.py` | ~200 LOC | `AutoMLOptimizer` — Optuna TPE sampler + MedianPruner. Optimerar: mode, batch_size, chunk_size, embedding_model, graph_algorithm, damping, threshold, clustering_eps. NAS-prototyp. |
| `federated/learning.py` | ~150 LOC | `FederatedLearningPipeline` — PySyft virtual workers, FedAvg aggregation (viktat efter sample count), `evaluate_global_model()`. Privacy-preserving training. |

---

## Level 0 (Support): Cache, Export, XAI, Audit, Plugins

### `cache/redis_cache.py` [KEY] — Distribuerad cachning (~630 LOC)

Tre backend-klasser med gemensamt `CacheBackend` ABC-interface:

| Klass | Backend | Funktioner |
|-------|---------|------------|
| `RedisCache` | Redis (aioredis) | Async get/set/delete, TTL, auto-genererade nycklar (MD5), statistik |
| `MemcachedCache` | Memcached (aiomcache) | Samma interface, lättare alternativ |
| `FallbackCache` | Redis → Memcached | Automatisk fallback: provar primary, vid fel → secondary. Graceful degradation. |

Alla operationer returnerar `None`/`False` vid fel istället för att kasta exceptions.

### `cache/manager.py` — In-memory LRU-cache

`CacheManager` — Enkel OrderedDict-baserad LRU med konfigurerbar max_size. 31 LOC. Används av engine som intern cache.

### `explainability/xai.py` [KEY] — Förklarbar AI (~785 LOC)

`ExplainableExtractor` — Förklara **varför** ett nyckelord extraherades:

- **6 förklaringskomponenter:**
  1. Lingvistisk analys (NER, POS, termfrekvens)
  2. Semantisk viktighet (document coherence, grannord, kluster)
  3. Graf-centralitet (degree, betweenness, closeness, PageRank)
  4. Kontextanalys (position i dokument, syntaktisk roll, co-occurrence)
  5. LIME-förklaring (feature importance via perturbation)
  6. Counterfactual-analys (vad händer om nyckelordet tas bort?)

- **`KeywordExplanation`** — Komplett förklaring med decision path, alternativa nyckelord, confidence

`ExplanationVisualizer` — Plotly-baserade visualiseringar:
- Importance bar chart
- Evidence sunburst
- Decision path diagram
- Komplett HTML-rapport (`create_explanation_report()`)

### `audit/lineage.py` — Spårbarhet & GDPR

- **`AuditLog`** — SQLAlchemy-modell: event_type, user_id, session_id, resource_id, action, status, duration_ms, ip_address. Dubbla index (user+time, resource).
- **`DataLineage`** — NetworkX-grafbaserad data lineage: spårar input → process → output → model → cache.
- **`AuditEventType`** — 12 eventtyper (extraction, model, cache, security, data access/modification).

### `export/formats.py` — Exporthantering

`ExportManager` — 4 format:

| Metod | Output |
|-------|--------|
| `to_json()` | Standard JSON med keywords + metadata |
| `to_csv()` | Flat CSV via pandas DataFrame |
| `to_graphml()` | Semantisk graf för Gephi/yEd |
| `to_embeddings()` | Raw vektorer dict för Pinecone/Weaviate |

### `plugins/system.py` — Pluginsystem (~440 LOC)

`PluginManager` — Komplett plugin-livscykel:

- **Discovery:** Scannar plugin-katalog, extraherar YAML-metadata från docstrings
- **Registrering:** `register_plugin()` kategoriserar som Extractor eller Processor
- **Hooks:** `register_hook()` / `execute_hook()` — async-kompatibelt eventsystem
- **Exempelplugins:** `DomainSpecificExtractor`, `AcademicCitationProcessor`
- **Plugin-typer:** `ExtractorPlugin` (custom nyckelordsextraktion), `ProcessorPlugin` (post-processing)

### `monitoring/metrics.py` — Prometheus-metrics

6 fördefinierade Prometheus-metrics:

| Metric | Typ | Beskrivning |
|--------|-----|-------------|
| `sie_x_requests_total` | Counter | Requests per metod/endpoint/status |
| `sie_x_request_duration_seconds` | Histogram | Latens per endpoint |
| `sie_x_errors_total` | Counter | Fel per typ |
| `sie_x_active_requests` | Gauge | Aktiva requests just nu |
| `sie_x_keywords_extracted_total` | Counter | Totalt extraherade nyckelord |
| `sie_x_model_info` | Gauge | Laddade modeller |

`get_metrics_app()` returnerar ASGI-app som mountas på `/metrics`.

Legacy-wrapper: `MetricsCollector` med timers och interna stats.

---

## Dokumentation & Exempel

| Fil | Syfte |
|-----|-------|
| `FULL_SYSTEM_ARCHITECTURE.md` | Komplett teknisk referens — alla moduler, klasser, metoder |
| `README.md` | Projektöversikt, snabbstart, modulöversikt |
| `SIEX_CAPABILITIES.md` | AI-agent capability guide — vad du kan göra med SIE-X |
| `PACKAGE_CONTENTS.md` | Denna fil — komplett filinventering |
| `arcitechture.txt` | Interna dataflödesnoteringar (legacy) |
| `core/interfaces.md` | Interfacekontrakt mellan core och övriga moduler |
| `core/SEI_X_CORE_README.md` | Core-specifik README |
| `api/README.md` | API-modulens dokumentation |
| `sdk/README.md` | SDK-dokumentation (standard + enterprise) |
| `examples/backlink_workflow.py` | Körbart SEO backlink-exempel: analys → bryggor → constraints |
| `prompts/writer_prompt.md` | BACOWR systemprompt för AI-skribent |

---

## Root Utilities

| Fil | Syfte |
|-----|-------|
| `project_builder.py` | Skapar en zip-fil med komplett SIE-X projektstruktur |
| `project_packager.py` | Samlar alla Python-filer i ett distribuerbart zip-arkiv |
| `usecase.py` | Körbara use case-exempel: juridik, medicin, hybrid finans-juridik |

---

## Beroenden per nivå

### Level 0: Core (minimum)
```
spacy, sentence-transformers, pydantic, networkx, numpy, torch, faiss-cpu
```

### Level 1–2: Transformers + API + Auth
```
fastapi, uvicorn, httpx, beautifulsoup4, python-jose, passlib, scikit-learn
```

### Level 3: Enterprise + Streaming + Orchestration
```
authlib, python3-saml, ldap3, redis, aiomcache, aiokafka, msgpack, websockets
prometheus-client, opentelemetry-api, opentelemetry-sdk, structlog
backoff, circuitbreaker, slowapi, langchain, llama-index
```

### Level 4: Autonomy + Self-optimization
```
ray[serve], optuna, syft, scipy, shap, lime, plotly
```

---

## Statistik

| Mått | Värde |
|------|-------|
| Modulkataloger | 22 |
| Python-källfiler | 38 |
| Dokumentationsfiler | 9 |
| Totalt LOC (uppskattat) | ~8 500 |
| Domäntransformers | 5 |
| API-endpoints | 14 |
| Prometheus-metrics | 6 |
| Stödda språk (core) | 11 |
| Stödda språk (multilingual) | 100+ |
| Modlägen | 4 (FAST, BALANCED, ADVANCED, ULTRA) |

**Licens:** Internt bruk / Proprietär.
