# SIE-X: Semantic Intelligence Engine X

**Version:** 2.0.0
**Arkitektur:** Modular Monolith (Python)
**Komplett referens:** Se [FULL_SYSTEM_ARCHITECTURE.md](FULL_SYSTEM_ARCHITECTURE.md) for full teknisk dokumentation av varje fil.

SIE-X är en produktionsredo motor for semantisk intelligens, nyckelords-extraktion, och domänspecifik analys. Den erbjuder:

- **Två motorer** — GPU-accelererad produktionsmotor (`engine.py`) och lättvikts-CPU-motor (`simple_engine.py`)
- **5 domäntransformers** — SEO, Medicin, Juridik, Finans, Kreativt skrivande
- **REST API** med auth, rate limiting, streaming
- **Python SDK** (standard + enterprise med WebSocket)
- **100+ språk** via LaBSE + XLM-RoBERTa
- **Självoptimering** via Ray-agenter, Optuna AutoML, A/B-testning, Active Learning
- **LangChain-integration** som drop-in replacement

---

## Snabbstart

### Installation

SIE-X är designad för att ligga som en undermapp i ditt projekt.

```bash
# Minimal (Core + API)
pip install fastapi uvicorn spacy sentence-transformers pydantic networkx numpy torch faiss-cpu

# + SEO & Transformers
pip install python-jose passlib httpx beautifulsoup4

# + Enterprise Auth
pip install authlib python3-saml ldap3 redis

# + Streaming & Observability
pip install aiokafka msgpack prometheus-client opentelemetry-api opentelemetry-sdk structlog

# + Autonomy (Agenter, AutoML, Federated Learning)
pip install ray[serve] optuna langchain llama-index syft

# Ladda ner språkmodell
python -m spacy download en_core_web_sm
```

### Hello World (Lättviktsmotor)

```python
from sie_x.core.simple_engine import SimpleSemanticEngine

engine = SimpleSemanticEngine()
keywords = engine.extract("Neural networks optimize semantic search.", top_k=5)

for kw in keywords:
    print(f"{kw.text}: {kw.score:.2f}")
```

### Hello World (Produktionsmotor)

```python
import asyncio
from sie_x.core.engine import SemanticIntelligenceEngine, ModelMode

engine = SemanticIntelligenceEngine(mode=ModelMode.BALANCED, enable_gpu=True)

keywords = asyncio.run(engine.extract_async(
    "Neural networks optimize semantic search.",
    top_k=10,
    output_format='json',
    enable_clustering=True
))
print(keywords)
```

### Hello World (API-klient)

```python
from sie_x.sdk.python.client import SIEXClient

client = SIEXClient("http://localhost:8000")
keywords = client.extract_sync("Neural networks optimize semantic search.")
for kw in keywords:
    print(f"{kw['text']}: {kw['score']:.2f}")
```

---

## Arkitektur — 5 lager

```
┌─────────────────────────────────────────────────────────────────┐
│  Level 4: AUTONOMY                                              │
│  agents/ · automl/ · testing/ · training/ · federated/          │
├─────────────────────────────────────────────────────────────────┤
│  Level 3: ORCHESTRATION                                         │
│  orchestration/ · streaming/ · sdk/                             │
├─────────────────────────────────────────────────────────────────┤
│  Level 2: INTEGRATIONS                                          │
│  api/ · auth/ · integrations/                                   │
├─────────────────────────────────────────────────────────────────┤
│  Level 1: TRANSFORMERS                                          │
│  transformers/seo · medical · legal · financial · creative      │
├─────────────────────────────────────────────────────────────────┤
│  Level 0: CORE ENGINE                                           │
│  core/ · chunking/ · cache/ · monitoring/ · export/             │
│  explainability/ · audit/ · plugins/ · multilingual/            │
└─────────────────────────────────────────────────────────────────┘
```

Varje nivå bygger på den under. Du kan använda bara Level 0 för enkel extraktion, eller hela stacken för enterprise-grade AI.

---

## Modulöversikt

### Core Engine (`core/`)

Hjärtat i systemet. Två motorer:

| Motor | Fil | GPU | Async | FAISS | Användning |
|-------|-----|-----|-------|-------|------------|
| **Production** | `engine.py` | Ja | Ja | Ja | Produktion, stora volymer |
| **Simple** | `simple_engine.py` | Nej | Nej | Nej | Utveckling, tester, lättvikt |

Övriga core-filer:

| Fil | Syfte |
|-----|-------|
| `models.py` | Pydantic-modeller (Keyword, Request, Response) — används överallt |
| `extractors.py` | CandidateExtractor + TermFilter — kandidatgenerering och brusfiltrering |
| `resilience.py` | Circuit breaker, exponential backoff, resurshantering |
| `streaming.py` | StreamingExtractor — chunked processing för stora dokument |
| `multilang.py` | 11 språk med auto-detektering (FastText) |
| `utils.py` | spaCy-modell auto-nedladdning |

### Domäntransformers (`transformers/`)

5 transformers som **injicerar** domänspecifik intelligens i core-motorn:

| Transformer | Fil | Transformerar till | Nyckelförmågor |
|-------------|-----|-------------------|----------------|
| **SEO** | `seo_transformer.py` | SEO-analys | Bridge topics, sökintention, content gaps |
| **Medicin** | `medical_transformer.py` | MedicalAI-X | Differentialdiagnos, läkemedelsinteraktioner, SOAP-anteckningar |
| **Juridik** | `legal_transformer.py` | LegalAI-X | SFS/EU-mönster, jurisdiktionshierarki, konfliktdetektering |
| **Finans** | `financial_transformer.py` | FinanceAI-X | Sentiment, trading signals, riskanalys |
| **Kreativt** | `creative_transformer.py` | CreativeAI-X | Narrativstruktur, karaktärsbågar, plot twists |

**Dynamisk laddning** via `loader.py`:

```python
from sie_x.transformers.loader import TransformerLoader

loader = TransformerLoader(engine)
loader.load_transformer('medical')           # Ladda en
loader.create_hybrid_system(['legal', 'financial'])  # Eller flera samtidigt
```

### REST API (`api/`)

FastAPI-server med full middleware-stack:

| Endpoint | Metod | Beskrivning |
|----------|-------|-------------|
| `/extract` | POST | Extrahera nyckelord från text |
| `/extract/batch` | POST | Batchbearbetning (bakgrundsjobb) |
| `/extract/stream` | GET | Server-Sent Events streaming |
| `/analyze/multi` | POST | Korsanalys mellan dokument |
| `/api/v1/analyze/url` | POST | Hämta URL och extrahera |
| `/api/v1/analyze/file` | POST | Ladda upp fil och extrahera |
| `/api/v1/keywords/search` | GET | Sök bland extraherade nyckelord |
| `/health` | GET | Hälsokontroll |

**Middleware:** JWT/API-key auth, rate limiting (100/h + burst 10/min), request tracing med `X-Request-ID`.

Starta servern:
```bash
uvicorn sie_x.api.server:app --host 0.0.0.0 --port 8000
```

### Enterprise Auth (`auth/`)

Tre SSO-protokoll i en unified manager:

| Protokoll | Bibliotek | Providers |
|-----------|-----------|-----------|
| OIDC | `authlib` | Azure AD, Okta, Auth0 |
| SAML | `python3-saml` | OneLogin, generisk |
| LDAP | `ldap3` | Active Directory, OpenLDAP |

Plus `TokenManager` med Redis-backed JWT-skapande, validering och revokering.

### Python SDK (`sdk/python/`)

Två klienter:

| Klient | Fil | Användning |
|--------|-----|------------|
| **Standard** | `client.py` | Async/sync, retry, URL/fil-analys |
| **Enterprise** | `sie_x_sdk.py` | + WebSocket streaming, OAuth, batch processor, caching |

### Chunking (`chunking/`)

Intelligent dokumentsplittning:
- `DocumentChunker` — Token-baserad med meningsgrits-respekt och konfigurerbar overlap.
- `SlidingWindowChunker` — Adaptiv fönsterstorlek baserad på informationsdensitet.

### Streaming (`streaming/`)

Real-time processing via:
- **Kafka** (`StreamingPipeline`) — 4 parallella workers, micro-batching, Redis-cache, dead letter queue.
- **WebSocket** (`WebSocketStreaming`) — Chunk-för-chunk resultat i realtid.

### Caching (`cache/`)

Två nivåer:
- `CacheManager` — In-memory LRU (OrderedDict) för utveckling och som engine-intern cache.
- `RedisCache` / `MemcachedCache` / `FallbackCache` — Distribuerad caching med graceful degradation.

### Monitoring (`monitoring/`)

| Fil | Omfattning |
|-----|-----------|
| `metrics.py` | Grundläggande Prometheus-counters |
| `observability.py` | Full stack: structlog (JSON) + Prometheus + OpenTelemetry tracing (Jaeger/Tempo) |

### Övriga moduler

| Modul | Fil | Syfte |
|-------|-----|-------|
| **Explainability** | `explainability/xai.py` | LIME, counterfactuals, Plotly-visualiseringar |
| **Audit** | `audit/lineage.py` | Data lineage (NetworkX), GDPR/CCPA-loggning (SQLAlchemy) |
| **Export** | `export/formats.py` | JSON, CSV, GraphML, raw embeddings |
| **Plugins** | `plugins/system.py` | Plugin-discovery, hooks, ExtractorPlugin/ProcessorPlugin |
| **Agenter** | `agents/autonomous.py` | Ray Serve: Monitor, Analyzer, Optimizer, Validator |
| **AutoML** | `automl/optimizer.py` | Optuna hyperparameter-tuning + NAS-prototyp |
| **Orchestration** | `orchestration/langchain_integration.py` | SIEXEmbeddings, SIEXRetriever, SIEXVectorStore |
| **Integrations** | `integrations/bacowr_adapter.py` | Backlink-automation: intent alignment, smart constraints |
| **Multilingual** | `multilingual/engine.py` | 100+ språk via LaBSE + XLM-RoBERTa |
| **Federated** | `federated/learning.py` | Privacy-preserving training med PySyft + FedAvg |
| **A/B Testing** | `testing/ab_framework.py` | Thompson sampling, statistisk analys, auto-deploy |
| **Active Learning** | `training/active_learning.py` | Feedback-loop, contrastive fine-tuning, uncertainty sampling |

---

## Use Cases

### Backlink Automation (BACOWR)

```python
from sie_x.transformers.seo_transformer import SEOTransformer
from sie_x.integrations.bacowr_adapter import BACOWRAdapter

# 1. Hitta pivot-topics
seo = SEOTransformer(engine)
bridges = seo.find_bridge_topics(publisher_text, target_text)

# 2. Generera smart constraints för AI-skribent
adapter = BACOWRAdapter(engine)
constraints = adapter.generate_smart_constraints(bridges, target_url)

# 3. (Valfritt) Förklara varför en anchor text valdes
from sie_x.explainability.xai import ExplainableExtractor
xai = ExplainableExtractor(engine)
explanation = xai.explain_extraction(text, "chosen anchor text")
```

Se `examples/backlink_workflow.py` och `prompts/writer_prompt.md`.

### RAG (Chat with Docs)

```python
from sie_x.orchestration.langchain_integration import SIEXRetriever

retriever = SIEXRetriever(engine)
# Använd som drop-in replacement i din LangChain-chain
```

### Medicinsk analys

```python
from sie_x.transformers.loader import TransformerLoader

loader = TransformerLoader(engine)
loader.load_transformer('medical')

result = await engine.extract_async(clinical_note)
# result innehåller: medical_entities, differential_diagnosis,
# drug_interactions, clinical_recommendations, risk_scores, SOAP note
```

---

## Root Utilities

| Fil | Syfte |
|-----|-------|
| `project_builder.py` | Automatiserar miljösetup och modellnedladdningar |
| `project_packager.py` | Paketerar `sie_x` som distribuerbar modul |
| `usecase.py` | Kör och testar specifika användarfall |
| `arcitechture.txt` | Interna dataflödesnoteringar |
| `SIEX_CAPABILITIES.md` | AI-agent capability guide |
| `PACKAGE_CONTENTS.md` | Legacy filinventering |

---

## Beroenden per nivå

### Level 0: Core (minimum)
```
spacy, sentence-transformers, pydantic, networkx, numpy, torch, faiss-cpu
```

### Level 1-2: Transformers + API
```
fastapi, uvicorn, httpx, beautifulsoup4, python-jose, passlib, scikit-learn
```

### Level 3: Enterprise
```
authlib, python3-saml, ldap3, redis, aiokafka, msgpack, websockets
prometheus-client, opentelemetry-api, opentelemetry-sdk, structlog
backoff, circuitbreaker, slowapi
```

### Level 4: Autonomy
```
ray[serve], optuna, langchain, llama-index, syft, scipy
```

---

## Statistik

- **26 modulkataloger**, **58 källfiler**, **6 dokumentationsfiler**
- Core engine: ~670 LOC (produktion), ~100 LOC (lättvikt)
- Domäntransformers: ~550 LOC (medicinsk), ~140 LOC (juridisk), ~165 LOC (finans), ~200 LOC (kreativ)
- A/B testing: ~450 LOC med Thompson sampling
- Stödjer 4 modlägen: FAST, BALANCED, ADVANCED, ULTRA

**Licens:** Internt bruk / Proprietär.
