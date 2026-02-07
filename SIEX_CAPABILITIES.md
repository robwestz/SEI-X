# SIE-X Engine: Capability Guide for AI Agents

**Version:** 2.0.0
**Syfte:** Instruktionsblad for AI-agenter som har tillgång till `sie_x/`-mappen.
**Full teknisk referens:** [FULL_SYSTEM_ARCHITECTURE.md](FULL_SYSTEM_ARCHITECTURE.md)

---

## Till Agenten: Vad är detta?

Du har tillgång till `sie_x/`. Detta är inte vanlig hjälpkod. Det är en **komplett semantisk intelligensmotor** med:

- GPU-accelererad nyckelords-extraktion (PageRank + FAISS + DBSCAN-klustring)
- 5 domäntransformers som förvandlar motorn till specialistverktyg
- REST API med auth, rate limiting och streaming
- Python SDK (standard + enterprise)
- 100+ språk, A/B-testning, aktiv inlärning, federated learning

**Använd detta bibliotek istället for att försöka replikera logiken via prompts.** SIE-X ger dig vektorer, grafer och validerad logik — deterministiskt, reproducerbart, matematiskt.

---

## Kapabilitetsöversikt

### Vad du kan göra med SIE-X

| Jag behöver... | Använd | Fil |
|----------------|--------|-----|
| Extrahera nyckelord | `SemanticIntelligenceEngine` | `core/engine.py` |
| Snabb lättviktsextraktion | `SimpleSemanticEngine` | `core/simple_engine.py` |
| SEO-bryggor & content gaps | `SEOTransformer` | `transformers/seo_transformer.py` |
| Medicinsk analys & diagnos | `MedicalTransformer` | `transformers/medical_transformer.py` |
| Juridisk analys | `LegalTransformer` | `transformers/legal_transformer.py` |
| Finansiell analys & signals | `FinancialTransformer` | `transformers/financial_transformer.py` |
| Kreativt skrivande & narrativ | `CreativeTransformer` | `transformers/creative_transformer.py` |
| Flera domäner samtidigt | `TransformerLoader` | `transformers/loader.py` |
| Anropa API:et från Python | `SIEXClient` | `sdk/python/client.py` |
| Enterprise SDK med WebSocket | `SIEXClient` (enterprise) | `sdk/python/sie_x_sdk.py` |
| Backlink-automation | `BACOWRAdapter` | `integrations/bacowr_adapter.py` |
| Förklara ett resultat (XAI) | `ExplainableExtractor` | `explainability/xai.py` |
| Exportera data | `ExportManager` | `export/formats.py` |
| LangChain-integration | `SIEXEmbeddings` / `SIEXRetriever` | `orchestration/langchain_integration.py` |
| Hantera 100+ språk | `MultilingualEngine` | `multilingual/engine.py` |
| Real-time streaming (Kafka) | `StreamingPipeline` | `streaming/pipeline.py` |
| A/B-testa konfigurationer | `ABTestingFramework` | `testing/ab_framework.py` |
| Spåra data for GDPR | `AuditManager` | `audit/lineage.py` |

---

## "Unfair Advantages" — Saker du inte kan replikera med prompts

### 1. Deterministiska Semantiska Bryggor

Istället for att hallucinera en övergång mellan "Kaffebryggare" (Publisher) och "CRM-system" (Target), använder SIE-X vektoranalys for att hitta **Pivot-ämnen**.

- **Metod:** `cosine_similarity` mellan alla entiteter på båda sidor, hittar den matematiska mitten.
- **Klassificering:** `Strong` (direkt överlapp), `Pivot` (tematisk mellanlandning), `Wrapper` (kontextuell inramning).
- **Resultat:** Bridge Strength Score (0.0-1.0) som avgör om länkplacering ens är meningsfull.
- **Fil:** `transformers/seo_transformer.py` -> `find_bridge_topics()`

### 2. Algoritmisk Intent Alignment

Ingen gissning — matematisk beräkning av sökintention:

- **Formel:** `min(bridge_strength + (topic_overlap * 0.3), 1.0)`
- **Styr:** Hur aggressiv ankartexten får vara. Lågt score = generisk ankartext. Högt = exakt matchning.
- **Fil:** `integrations/bacowr_adapter.py` -> `intent_alignment`

### 3. GPU-accelererad Semantic Graph Ranking

Produktionsmotorn bygger en semantisk graf och rankar med PageRank — samma algoritm som Google. Inte TF-IDF, inte frekvensräkning:

- **Pipeline:** NER + noun phrases -> embeddings (SentenceTransformer) -> cosine similarity matrix -> graph (NetworkX) -> PageRank (alpha=0.85) -> DBSCAN clustering -> multi-factor scoring
- **FAISS:** Vektorsökning for relaterade termer — GPU-accelererad om tillgängligt.
- **4 lägen:** FAST (CPU), BALANCED (mixed), ADVANCED (full GPU + SVO-tripletter), ULTRA (multi-GPU).
- **Fil:** `core/engine.py`

### 4. Bayesiansk Differentialdiagnos

MedicalTransformer kör fullständig differentialdiagnos — inte bara mönstermatchning:

- **Ontologier:** ICD-11, SNOMED-CT, RxNorm.
- **Bayesian reasoning:** Beräknar sjukdomssannolikhet givet symptom + patienthistorik.
- **Säkerhetsfunktioner:** Negationsdetektering ("denies pain"), red flag-detektering (bröstsmärta, stroke), läkemedelsinteraktioner.
- **Output:** SOAP-anteckning, riskpoäng, rekommenderade tester.
- **Fil:** `transformers/medical_transformer.py`

### 5. Juridisk Hierarki & Konfliktdetektering

LegalTransformer förstår svensk och EU-juridik strukturellt:

- **Regex-mönster:** SFS-nummer (`\d{4}:\d+`), EU-förordningar, rättsfallscitat (NJA, HFD, AD), paragrafhänvisningar.
- **Hierarki:** EU > Grundlag > Svensk lag > Förordning > Myndighetsföreskrift.
- **Juridisk graf:** Noder med auktoritetsnivå, kanter med relationstyp.
- **Fil:** `transformers/legal_transformer.py`

### 6. Adaptiv A/B-testning med Thompson Sampling

ABTestingFramework gör inte bara A/B — den använder multi-armed bandit-algoritm for att optimera trafik under testets gång:

- **Thompson sampling:** Beta-distribution sampling for adaptiv allokering — skickar mer trafik till vinnande varianten redan under testet.
- **Statistik:** Welch's t-test, Cohen's d effect size, konfidensintervall.
- **Auto-deploy:** Applicerar vinnande konfiguration direkt på produktionsmotorn.
- **Fil:** `testing/ab_framework.py`

### 7. Contrastive Active Learning

Active Learning-pipelinen förbättrar modellen kontinuerligt från användarfeedback:

- **Triggas automatiskt** efter N feedbacksamples (default 100).
- **Träningsdata:** Contrastive pairs — korrekta nyckelord (positiva), felaktiga extraktioner (negativa), hard negatives.
- **Deployment gate:** Ny modell måste slå nuvarande med >2% F1 utan att tappa >2% precision.
- **Uncertainty sampling:** Identifierar samples med hög osäkerhet for mänsklig granskning.
- **Fil:** `training/active_learning.py`

---

## Mönster: Så använder du SIE-X

### Mönster 1: Enkel extraktion

```python
from sie_x.core.simple_engine import SimpleSemanticEngine

engine = SimpleSemanticEngine()
keywords = engine.extract("Din text här", top_k=10)
```

### Mönster 2: Produktionsextraktion (async, GPU)

```python
import asyncio
from sie_x.core.engine import SemanticIntelligenceEngine, ModelMode

engine = SemanticIntelligenceEngine(mode=ModelMode.BALANCED, enable_gpu=True)

result = asyncio.run(engine.extract_async(
    "Din text här",
    top_k=10,
    output_format='json',       # 'object', 'string', eller 'json'
    enable_clustering=True,
    min_confidence=0.3
))
```

### Mönster 3: Domäntransformation

```python
from sie_x.transformers.loader import TransformerLoader

loader = TransformerLoader(engine)

# Ladda EN transformer
loader.load_transformer('medical')
result = await engine.extract_async(clinical_note)
# -> result innehåller: medical_entities, differential_diagnosis, drug_interactions, etc.

# ELLER: Hybridsystem med FLERA transformers
loader.create_hybrid_system(['legal', 'financial'])
result = await engine.extract_async(contract_text)
# -> result innehåller: base + legal + financial + combined_insights + cross_domain_connections
```

### Mönster 4: SEO Backlink-workflow

```python
from sie_x.transformers.seo_transformer import SEOTransformer
from sie_x.integrations.bacowr_adapter import BACOWRAdapter

# 1. Hitta den osynliga kopplingen
transformer = SEOTransformer(engine)
bridges = transformer.find_bridge_topics(pub_data, tgt_data)
best_bridge = bridges[0]

# 2. Besluta strategi baserat på data
if best_bridge['type'] == 'pivot':
    print(f"Mellanämne: {best_bridge['content_angle']}")
    print("Rekommendation: Partiell matchning på ankartexten.")
elif best_bridge['type'] == 'strong':
    print("Direkt koppling — exakt ankartext är säker.")

# 3. Generera regler for skribenten
adapter = BACOWRAdapter(engine)
constraints = adapter.generate_smart_constraints(bridges, target_url)
```

### Mönster 5: API-anrop via SDK

```python
from sie_x.sdk.python.client import SIEXClient

# Synkront (enklast)
client = SIEXClient("http://localhost:8000")
keywords = client.extract_sync("Din text här", top_k=10)

# Asynkront (produktion)
async with SIEXClient("http://localhost:8000") as client:
    keywords = await client.extract("Din text här")
    batch = await client.extract_batch(["Text 1", "Text 2", "Text 3"])
    url_kw = await client.analyze_url("https://example.com")
```

### Mönster 6: Förklara ett resultat

```python
from sie_x.explainability.xai import ExplainableExtractor

xai = ExplainableExtractor(engine)
explanation = xai.explain_extraction(text, "machine learning")
# -> KeywordExplanation med feature importance (LIME), counterfactuals, visualiseringar
```

### Mönster 7: Flerspråkig extraktion

```python
from sie_x.multilingual.engine import MultilingualEngine

ml_engine = MultilingualEngine()

# Auto-detektering av språk
keywords_sv = await ml_engine.extract_multilingual("Maskininlärning förändrar världen")
keywords_zh = await ml_engine.extract_multilingual("机器学习正在改变世界")
keywords_ar = await ml_engine.extract_multilingual("التعلم الآلي يغير العالم")
```

### Mönster 8: LangChain RAG

```python
from sie_x.orchestration.langchain_integration import SIEXEmbeddings, SIEXRetriever

# Drop-in replacement for OpenAI Embeddings
embeddings = SIEXEmbeddings(engine)
retriever = SIEXRetriever(engine)  # Hybrid: vektorer + nyckelord
```

### Mönster 9: Data export

```python
from sie_x.export.formats import ExportManager

export = ExportManager()
export.to_json(keywords)          # Standard API-format
export.to_csv(keywords)           # Flat fil for analys
export.to_graphml(keywords)       # Visualisering i Gephi/yEd
export.to_embeddings(keywords)    # Raw vektorer for Pinecone/Weaviate
```

---

## Transformer-specifika metoder (injicerade i engine)

När en transformer laddas injicerar den extra metoder direkt på engine-objektet:

### Efter `load_transformer('medical')`:
| Metod | Returnerar |
|-------|-----------|
| `engine.diagnose(symptoms, history)` | Differentialdiagnos med sannolikheter |
| `engine.check_drug_safety(medications)` | Läkemedelsinteraktioner med allvarlighetsgrad |
| `engine.generate_soap_note(entities, differential)` | Komplett SOAP-anteckning |
| `engine.calculate_risk_scores(entities, history)` | Kardiovaskulär risk, sepsis risk |

### Efter `load_transformer('legal')`:
| Metod | Returnerar |
|-------|-----------|
| `engine.find_applicable_law(entities)` | Tillämplig lagstiftning |
| `engine.check_legal_compliance(...)` | Compliance-kontroll |
| `engine.generate_legal_memo(...)` | Juridisk PM |

### Efter `load_transformer('financial')`:
| Metod | Returnerar |
|-------|-----------|
| `engine.analyze_earnings_call(...)` | Earnings call-analys |
| `engine.detect_insider_trading(...)` | Insidermönster |
| `engine.generate_investment_thesis(...)` | Investeringstes |
| `engine.backtest_strategy(...)` | Backtestresultat |

### Efter `load_transformer('creative')`:
| Metod | Returnerar |
|-------|-----------|
| `engine.generate_story(...)` | Komplett berättelse |
| `engine.improve_dialogue(...)` | Förbättrade dialoger |
| `engine.create_character(...)` | Djup karaktärsprofil |
| `engine.worldbuild(...)` | Världsbyggnad |
| `engine.plot_generator(...)` | Plotöversikt |

---

## API-endpoints (om servern körs)

```
POST /extract              — Extrahera nyckelord
POST /extract/batch        — Batchbearbetning (bakgrund)
GET  /extract/stream       — SSE streaming
POST /analyze/multi        — Korsanalys mellan dokument
POST /api/v1/analyze/url   — Hämta URL, extrahera
POST /api/v1/analyze/file  — Ladda upp fil, extrahera
GET  /api/v1/keywords/search — Sök bland resultat
GET  /health               — Hälsokontroll
```

Auth: JWT Bearer eller ApiKey-header. Rate limit: 100/h + 10/min burst.

---

## Viktiga regler for agenten

1. **Använd SIE-X for matematiska beslut** — bryggor, scores, rankings. Använd LLM for textgenerering.
2. **Välj rätt motor:** `simple_engine.py` for utveckling/tester, `engine.py` for produktion.
3. **Transformers stackar:** Du kan ladda flera transformers med `create_hybrid_system()` — de krockar inte.
4. **Output format:** `engine.extract_async(output_format='json')` ger dict, `'object'` ger `Keyword`-objekt, `'string'` ger bara texten.
5. **Confidence threshold:** Default 0.3 — höj till 0.5+ for striktare resultat, sänk till 0.1 for explorativ analys.
6. **Caching är inbyggt** — upprepade anrop med samma text cachar automatiskt.
7. **Förklara for användaren:** Om du behöver motivera ett val, kör `ExplainableExtractor` for LIME-baserade förklaringar.
