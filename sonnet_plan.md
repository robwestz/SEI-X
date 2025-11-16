SIE-X STATUS & SMART FASPLAN
1. ÄRLIG BEDÖMNING AV NULÄGET
Efter att ha gått igenom all kod vi skapat:
Production-Ready (70-90% klart):

Terraform/K8s deployment configs
Docker/CI-CD struktur
SDK interfaces (Python/Node/Go)
Datamodeller (Keyword, ExtractionOptions)

Halvfärdigt (30-70% klart):

Core engine struktur (har arkitektur men saknar kärnlogik)
API endpoints (definierade men ej implementerade)
Basic chunking/caching struktur
Transformer koncept (bra design, saknar implementation)

Vision/Boilerplate (0-30% klart):

Active learning loop
Autonomous agents
Federated learning
Multi-language support detaljer
XAI visualiseringar

2. SMART FASPLAN
mermaidgraph TD
    A[Fas 1: Minimal Viable Engine] --> B[Fas 2: Transformers & Features]
    B --> C[Fas 3: SEO/Backlink Product]
    
    A --> A1[Core Extract]
    A --> A2[REST API]
    A --> A3[Python SDK]
    A --> A4[Docker Setup]
    
    B --> B1[Legal/Medical/Financial]
    B --> B2[Streaming]
    B --> B3[Multi-lang]
    
    C --> C1[SEO Transformer]
    C --> C2[Backlink API]
    C --> C3[Bulk Processor]
FAS 1: Minimal Körbar Kärna (1-2 veckor)
Mål: En fungerande keyword extraction API som kan köras lokalt
Tier 1 Filer:

sie_x/core/simple_engine.py - Förenklad motor med SBERT + spaCy
sie_x/api/minimal_server.py - FastAPI med 3 endpoints
sie_x/sdk/python/client.py - Enkel Python-klient
docker-compose.minimal.yml - Redis + API
demo/quickstart.py - Demo som visar värdet

FAS 2: Power Features (3-4 veckor)
Mål: Aktivera transformers och avancerade features
Prioritet:

SEO Transformer (för Fas 3)
Streaming support
Basic caching/monitoring
Multi-document analysis

FAS 3: SEO/Backlink Intelligence (2-3 veckor)
Mål: En säljbar produkt för SEO-byråer
3. TIER 1 MODULER
pythonTIER_1_MODULES = {
    "Core": [
        "sie_x/core/simple_engine.py",      # Huvudmotor
        "sie_x/core/models.py",              # Datamodeller
        "sie_x/core/extractors.py",          # Keyword extraction
    ],
    "API": [
        "sie_x/api/minimal_server.py",       # FastAPI server
        "sie_x/api/routes.py",               # API routes
    ],
    "SDK": [
        "sie_x/sdk/python/client.py",        # Python client
    ],
    "Infra": [
        "docker-compose.minimal.yml",        # Docker setup
        "requirements.minimal.txt",          # Dependencies
    ]
}
4. SONNET PROMPT PACK - FAS 1
PROMPT 1: Core Simple Engine
markdownFOCUS FILE: sie_x/core/simple_engine.py

CONTEXT: You're implementing the core extraction engine for SIE-X (Semantic Intelligence Engine). This is a SIMPLIFIED version for Fas 1 - we need it working, not perfect.

EXISTING STRUCTURE:
- Models defined in models.py (Keyword, ExtractionOptions)
- Will be called by API in minimal_server.py
- Should use sentence-transformers and spaCy

REQUIREMENTS:
1. Class: SimpleSemanticEngine
2. Main method: extract(text: str, top_k: int = 10) -> List[Keyword]
3. Use 'sentence-transformers/all-MiniLM-L6-v2' (faster model)
4. Basic candidate generation with spaCy NER + noun chunks
5. Simple graph-based ranking (PageRank on similarity graph)
6. Return Keyword objects with: text, score, type, count

IMPLEMENTATION GUIDELINES:
- Keep it under 300 lines
- Use try/except for robustness
- Add logging
- Cache embeddings in memory (simple dict)
- Focus on WORKING over PERFECT

EXPECTED OUTPUT:
A complete, runnable simple_engine.py that can extract keywords from text using semantic similarity.
PROMPT 2: Minimal API Server
markdownFOCUS FILE: sie_x/api/minimal_server.py

CONTEXT: FastAPI server exposing the SimpleSemanticEngine. This is the Fas 1 minimal version.

EXISTING:
- SimpleSemanticEngine in core/simple_engine.py
- Models in core/models.py

REQUIREMENTS:
1. FastAPI app with 3 endpoints:
   - POST /extract - Extract keywords from text
   - GET /health - Health check
   - GET /models - List available models
2. Pydantic models for requests/responses
3. Global engine instance (singleton)
4. Basic error handling
5. CORS enabled for frontend

API SPEC:
POST /extract
Body: {"text": "...", "options": {"top_k": 10}}
Response: {"keywords": [...], "processing_time": 0.123}

IMPLEMENTATION:
- Under 200 lines
- Async endpoints
- Proper status codes
- Request validation
- Simple in-memory rate limiting (optional)

EXPECTED OUTPUT:
Complete minimal_server.py that can be run with: uvicorn sie_x.api.minimal_server:app
PROMPT 3: Python SDK Client
markdownFOCUS FILE: sie_x/sdk/python/client.py

CONTEXT: Simple Python client for the SIE-X API. Fas 1 version - minimal but functional.

REQUIREMENTS:
1. Class: SIEXClient
2. Methods:
   - __init__(base_url="http://localhost:8000")
   - extract(text: str, top_k: int = 10) -> List[Dict]
   - extract_batch(texts: List[str], top_k: int = 10) -> List[List[Dict]]
   - health_check() -> bool
3. Use httpx for async support
4. Basic retry logic (3 attempts)
5. Timeout handling (30s default)

USAGE EXAMPLE:
```python
client = SIEXClient()
keywords = await client.extract("Apple announced new iPhone")
print(keywords[0])  # {"text": "Apple", "score": 0.95, ...}
```

IMPLEMENTATION:
- Under 150 lines
- Clean error messages
- Type hints
- Docstrings

EXPECTED OUTPUT:
Complete client.py that works with the minimal API server.
PROMPT 4: Docker Compose Setup
markdownFOCUS FILE: docker-compose.minimal.yml

CONTEXT: Minimal Docker setup for running SIE-X locally in Fas 1.

REQUIREMENTS:
1. Services:
   - sie-x-api (the FastAPI server)
   - redis (for future caching)
2. Use Python 3.10 base image
3. Install requirements.minimal.txt
4. Download spacy model on build
5. Expose port 8000
6. Health checks

STRUCTURE:
```yaml
version: '3.8'
services:
  sie-x-api:
    build: ...
    ports: ...
    environment: ...
    depends_on: ...
  redis:
    image: redis:7-alpine
    ...
```

EXPECTED OUTPUT:
Complete docker-compose.minimal.yml + Dockerfile.minimal that runs with: docker-compose -f docker-compose.minimal.yml up
PROMPT 5: Core Models
markdownFOCUS FILE: sie_x/core/models.py

CONTEXT: Data models for SIE-X using Pydantic. These are used across engine, API, and SDK.

REQUIREMENTS:
1. Classes:
   - Keyword (text, score, type, count, confidence=1.0)
   - ExtractionOptions (top_k=10, min_confidence=0.3, language="en")
   - ExtractionRequest (text, options: Optional[ExtractionOptions])
   - ExtractionResponse (keywords: List[Keyword], processing_time: float, version: str)
   - BatchExtractionRequest (texts: List[str], options)
   - HealthResponse (status, version, models_loaded)

2. All models should:
   - Use Pydantic BaseModel
   - Have field validators where needed
   - Support JSON serialization
   - Have useful __str__ methods

IMPLEMENTATION:
- Under 100 lines
- Clear field descriptions
- Example in docstrings

EXPECTED OUTPUT:
Complete models.py with all data classes needed for Fas 1.
5. SEO TRANSFORMER DESIGN (För Fas 3)
python# Design sketch för SEO Transformer
class SEOTransformer:
    """Transformerar SIE-X till SEO/Backlink Intelligence Engine"""
    
    def transform_extraction(self, original_extract):
        async def seo_extract(text: str, url: str = None, **kwargs):
            # Original keywords
            keywords = await original_extract(text, **kwargs)
            
            # SEO-specifik analys
            seo_data = {
                "keywords": keywords,
                "entities": self._extract_entities(keywords),
                "topics": self._cluster_topics(keywords),
                "link_opportunities": self._find_link_opportunities(keywords),
                "content_gaps": self._identify_content_gaps(keywords),
                "serp_alignment": self._check_serp_alignment(keywords, url),
                "anchor_suggestions": self._generate_anchors(keywords)
            }
            
            return seo_data
NÄSTA STEG

Skapa repo-struktur för Fas 1 filerna
Kör Sonnet-prompts i ordning (1-5)
Testa med Docker så snart grundfilerna är klara
Iterera baserat på vad som fungerar

Vill du att jag:

Generera fler Sonnet-prompts för remaining Tier 1 filer?
Detaljera SEO Transformer designen mer?
Skapa en "Quick Win Demo" som visar värdet?
RetryRInnan jag skickar det chatgpt tyckte att jag skulle svara så kan jag bara säga att jag har krediter i claude kod browser som räcker typ en dag till men jag har  krediter för flera 100k LOC, så jag vill verkligen nyttja det, ha det i åtanke Din fasplan (Fas 1–3) känns helt rätt, och jag vill nu börja exekvera den på ett sätt som funkar ihop med mitt befintliga repo.
1. Kör Fas 1 utifrån mitt nuvarande repo
Repo: https://github.com/robwestz/SEI-X
Jag vill att Fas 1 implementeras så att den:
* Utgår från befintlig struktur i repot (mappnamn, sie_x/core, sie_x/api, sie_x/sdk/python osv).
* Skapar en minimal körbar kedja:
   * SimpleSemanticEngine → minimal FastAPI-server → Python-SDK → Docker-compose.
* Är helt körbar lokalt (uvicorn + docker-compose) utan att jag behöver fylla i en massa själv.
2. Ja, jag vill ha fler Sonnet-prompts – men för ALLA Tier 1-delar
Jag vill att du:
1. Tar listan med TIER_1_MODULES du föreslog och:
   * Mappar den 1:1 mot det faktiska repo: sie_x/core/simple_engine.py, models.py, extractors.py, sie_x/api/minimal_server.py, routes.py, sie_x/sdk/python/client.py, docker-compose.minimal.yml, requirements.minimal.txt, ev. demo/quickstart.py.
2. För varje sådan fil:
   * Skapar en dedikerad Sonnet-prompt i stil med de fem du redan skrev (FOCUS FILE, CONTEXT, REQUIREMENTS, IMPLEMENTATION, EXPECTED OUTPUT).
   * Skriver dem så konkret att Sonnet kan spotta ut färdig, körbar kod utan följdfrågor.
3. Packar alla dessa prompts i ett snyggt “Prompt Pack Fas 1”, t.ex. som en numrerad lista jag kan copy-pasta fil för fil in i Sonnet.
3. SEO/Backlink-fokus – kopplingen till BACOWR
Jag har redan en ganska avancerad backlink-pipeline (BACOWR) med:
* run_backlink_job / run_production_job
* Profilering (publisher/target)
* Ahrefs SERP-research
* Intent-analys
* Writer + QC + AutoFix
Målet är att SIE-X i Fas 3 ska bli semantic core för detta.
Så, jag vill också att du:
1. Detaljerar SEO Transformer-designen mer, men specifikt i relation till BACOWR:
   * Vilka fält i BacklinkJobPackage som ska komma från SIE-X (entities, topics, kluster, content gaps, anchor-suggestions, link-opps etc).
   * Vilka metoder SEO-transformern behöver (t.ex. analyze_publisher, analyze_target, analyze_serp, suggest_bridges, score_anchor_risk).
   * Hur SEO-transformern ska “hookas in” i befintliga steg:
      * PageProfiler
      * IntentAnalyzer
      * serp_research_ext
      * generation_constraints.
2. Ger mig en konkret filstruktur för Fas 3 SEO-transformer, t.ex.:
   * sie_x/transformers/seo_transformer.py
   * sie_x/integrations/bacowr_adapter.py
   * plus ev. nya modeller om det behövs.
4. Quick Win Demo – ja tack
Jag vill också ha en Quick Win Demo som jag kan visa upp för folk:
* Fas 1-baserad (minimal engine + API), men gärna redan tänkt med SEO/keyword-användning i åtanke.
* Förslag:
   * En liten demo (demo/quickstart.py + ev. enkel frontend) där man:
      * klistrar in text eller URL,
      * får tillbaka keywords + topics,
      * ser att det körs via SIE-X-API:t.
* Beskriv:
   * vilka filer som behövs,
   * hur de hänger ihop,
   * exakt vilka kommandon jag kör för att prova.
Kort sagt:
* ✅ Kör på din Fas 1-plan,
* ✅ Generera fullständigt Sonnet Prompt Pack för alla Tier 1-filer,
* ✅ Fördjupa SEO Transformer-designen med fokus på integration med min BACOWR-pipeline,
* ✅ Skissa en Quick Win Demo som gör det lätt att visa värdet.Show moreOrchestrated comprehensive Phase 1 execution plan with complete Sonnet prompts and BACOWR integration strategy.Användaren har:

Ett GitHub repo för SIE-X
Claude Code browser credits för en dag till + credits för flera 100k LOC
Vill maximera användningen av Claude Code för implementation
Har redan en BACOWR pipeline för backlinks som SIE-X ska integrera med
Vill ha konkreta Sonnet-prompts för ALLA Tier 1 filer
Vill ha en Quick Win Demo

Jag ska:

Skapa ett komplett Prompt Pack för Fas 1 med ALLA Tier 1 filer
Designa SEO Transformer med specifik koppling till BACOWR
Föreslå Quick Win Demo struktur
Vara konkret och implementation-fokuserad eftersom de har gott om Claude Code credits

Tier 1 filer att skapa prompts för:

sie_x/core/simple_engine.py
sie_x/core/models.py
sie_x/core/extractors.py
sie_x/api/minimal_server.py
sie_x/api/routes.py
sie_x/sdk/python/client.py
docker-compose.minimal.yml
requirements.minimal.txt
demo/quickstart.py
Dockerfile.minimal
SIE-X FAS 1 EXECUTION PLAN + CLAUDE CODE MAXIMERING
1. KOMPLETT SONNET PROMPT PACK - FAS 1
Här är ALLA prompts för Tier 1. Kör dem i ordning i Claude Code:
PROMPT 1/10: Core Models
markdownFOCUS FILE: sie_x/core/models.py

CONTEXT: Foundation data models for SIE-X. These are used everywhere - engine, API, SDK. Start here.

REQUIREMENTS:
```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# Required classes:
1. Keyword:
   - text: str
   - score: float (0-1)
   - type: str (NER type: PERSON, ORG, LOC, CONCEPT)
   - count: int = 1
   - confidence: float = 1.0
   - positions: List[tuple[int,int]] = []
   - metadata: Dict[str, Any] = {}

2. ExtractionOptions:
   - top_k: int = 10
   - min_confidence: float = 0.3
   - include_entities: bool = True
   - include_concepts: bool = True
   - language: str = "en"

3. ExtractionRequest:
   - text: str (max 10000 chars)
   - url: Optional[str] = None
   - options: Optional[ExtractionOptions] = None

4. ExtractionResponse:
   - keywords: List[Keyword]
   - processing_time: float
   - version: str = "1.0.0"
   - metadata: Dict[str, Any] = {}

5. BatchExtractionRequest:
   - items: List[ExtractionRequest]
   - options: Optional[ExtractionOptions] = None

6. HealthResponse:
   - status: str
   - version: str
   - models_loaded: List[str]
   - uptime: float
```

IMPLEMENTATION:
- Add JSON example in class docstring
- Field validators for score (0-1), confidence (0-1)
- Custom __str__ for Keyword
- Config for JSON by alias

EXPECTED OUTPUT: Complete models.py with all classes, ~150 lines
PROMPT 2/10: Simple Engine
markdownFOCUS FILE: sie_x/core/simple_engine.py

CONTEXT: The ACTUAL extraction engine. This is the heart of SIE-X Fas 1. Must be simple but effective.

EXISTING: models.py with Keyword class

REQUIREMENTS:
```python
import spacy
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple
import logging

class SimpleSemanticEngine:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Load models
        self.embedder = SentenceTransformer(model_name)
        self.nlp = spacy.load("en_core_web_sm")  # Start with small model
        self.embedding_cache = {}
        
    def extract(self, text: str, top_k: int = 10, min_confidence: float = 0.3) -> List[Keyword]:
        # 1. Generate candidates (NER + noun phrases)
        # 2. Create embeddings (with caching)
        # 3. Build similarity graph
        # 4. Run PageRank
        # 5. Return top-k Keywords
        
    def _generate_candidates(self, text: str) -> Dict[str, Keyword]:
        # Use spaCy for NER and noun chunks
        # Return dict of normalized_text -> Keyword
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        # Batch embed with caching
        
    def _build_graph(self, keywords: List[Keyword], embeddings: np.ndarray) -> nx.Graph:
        # Create similarity graph, threshold > 0.3
        
    def _rank_keywords(self, graph: nx.Graph, keywords: List[Keyword]) -> List[Keyword]:
        # PageRank + combine with frequency
```

IMPLEMENTATION DETAILS:
- Cache embeddings by text hash
- Handle empty text gracefully
- Log processing steps
- Keep under 250 lines
- Add docstrings

EXPECTED OUTPUT: Working simple_engine.py that extracts keywords using semantic similarity
PROMPT 3/10: Extractors Module
markdownFOCUS FILE: sie_x/core/extractors.py

CONTEXT: Additional extraction utilities and candidate generators. Supports simple_engine.py.

REQUIREMENTS:
```python
from typing import List, Set, Tuple
import re
from collections import Counter

class CandidateExtractor:
    """Generate keyword candidates from text"""
    
    def extract_entities(self, doc) -> List[Tuple[str, str]]:
        # Extract named entities from spaCy doc
        # Return [(text, label), ...]
    
    def extract_noun_phrases(self, doc) -> List[str]:
        # Extract meaningful noun phrases
        # Filter out pronouns, single determiners
    
    def extract_key_phrases(self, text: str) -> List[str]:
        # Regex-based extraction for patterns like:
        # - "X of Y" 
        # - "X and Y"
        # - Quoted phrases
    
    def normalize_text(self, text: str) -> str:
        # Lowercase, remove extra spaces
        # Keep punctuation that matters (. , -)

class TermFilter:
    """Filter out noise from candidates"""
    
    def __init__(self):
        self.min_length = 2
        self.max_length = 50
        self.stop_patterns = [r'^\d+$', r'^[^\w]+$']
    
    def is_valid(self, term: str) -> bool:
        # Check length, patterns, etc.
    
    def filter_candidates(self, candidates: List[str]) -> List[str]:
        # Apply all filters
```

IMPLEMENTATION:
- Make it modular for easy testing
- Handle edge cases (empty text, all numbers, etc.)
- ~150 lines total

EXPECTED OUTPUT: Complete extractors.py with utility functions for candidate generation
PROMPT 4/10: API Server
markdownFOCUS FILE: sie_x/api/minimal_server.py

CONTEXT: FastAPI server exposing SimpleSemanticEngine. This makes SIE-X accessible via HTTP.

EXISTING: 
- simple_engine.py with SimpleSemanticEngine
- models.py with all request/response models

REQUIREMENTS:
```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
from typing import Optional

from sie_x.core.simple_engine import SimpleSemanticEngine
from sie_x.core.models import *

app = FastAPI(title="SIE-X API", version="1.0.0")

# Global engine instance
engine: Optional[SimpleSemanticEngine] = None

@app.on_event("startup")
async def startup():
    # Initialize engine
    # Log startup
    
@app.post("/extract", response_model=ExtractionResponse)
async def extract_keywords(request: ExtractionRequest):
    # Validate request
    # Call engine.extract()
    # Handle errors
    # Return response with timing
    
@app.post("/extract/batch", response_model=List[ExtractionResponse])
async def extract_batch(request: BatchExtractionRequest):
    # Process multiple texts
    # Use asyncio for parallel processing
    
@app.get("/health", response_model=HealthResponse)
async def health_check():
    # Return status, version, uptime
    
@app.get("/models")
async def list_models():
    # Return available models and their status
```

MIDDLEWARE:
- CORS for frontend access
- Request ID tracking
- Basic rate limiting (10 req/second)
- Logging middleware

ERROR HANDLING:
- 400 for invalid input
- 413 for text too long
- 500 for engine errors
- Proper error messages

EXPECTED OUTPUT: Complete minimal_server.py, ~200 lines, runnable with uvicorn
PROMPT 5/10: API Routes
markdownFOCUS FILE: sie_x/api/routes.py

CONTEXT: Additional API routes and utilities. Separates routing logic from server setup.

REQUIREMENTS:
```python
from fastapi import APIRouter, Query, Path
from typing import List, Optional
import aiofiles

router = APIRouter(prefix="/api/v1", tags=["extraction"])

@router.post("/analyze/url")
async def analyze_url(url: str):
    # Fetch URL content
    # Extract text
    # Run extraction
    # Return results
    
@router.post("/analyze/file")
async def analyze_file(file: UploadFile):
    # Handle txt, pdf, docx
    # Extract text
    # Run extraction
    
@router.get("/keywords/search")
async def search_keywords(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100)
):
    # Search in extracted keywords (future feature)
    # For now, return mock data
    
@router.get("/stats")
async def get_stats():
    # Return API usage statistics
    # - Total extractions
    # - Average processing time
    # - Cache hit rate

# Utility functions
async def fetch_url_content(url: str) -> str:
    # Use httpx to fetch URL
    # Extract text from HTML
    
def extract_text_from_file(file: UploadFile) -> str:
    # Handle different file types
```

IMPLEMENTATION:
- Use httpx for async HTTP
- BeautifulSoup for HTML parsing
- Keep simple for Fas 1
- ~150 lines

EXPECTED OUTPUT: Complete routes.py with additional endpoints
PROMPT 6/10: Python SDK Client
markdownFOCUS FILE: sie_x/sdk/python/client.py

CONTEXT: Python client library for SIE-X API. Should be pip-installable and easy to use.

REQUIREMENTS:
```python
import httpx
from typing import List, Dict, Optional, Union
import asyncio
from urllib.parse import urljoin

from sie_x.core.models import (
    ExtractionRequest, ExtractionResponse,
    BatchExtractionRequest, Keyword
)

class SIEXClient:
    """Python client for SIE-X API"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        api_key: Optional[str] = None
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.api_key = api_key
        self._client = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, *args):
        await self._client.aclose()
    
    async def extract(
        self, 
        text: str,
        top_k: int = 10,
        min_confidence: float = 0.3,
        url: Optional[str] = None
    ) -> List[Dict[str, any]]:
        # Build request
        # POST to /extract
        # Parse response
        # Return keywords as dicts
    
    async def extract_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[Dict[str, any]]]:
        # Batch processing
        
    async def analyze_url(self, url: str, **kwargs) -> List[Dict[str, any]]:
        # POST to /analyze/url
        
    async def health_check(self) -> bool:
        # GET /health
        
    # Sync wrappers for convenience
    def extract_sync(self, text: str, **kwargs):
        return asyncio.run(self.extract(text, **kwargs))
```

USAGE EXAMPLE in docstring:
```python
# Async usage
async with SIEXClient() as client:
    keywords = await client.extract("Apple announced new iPhone")
    
# Sync usage
client = SIEXClient()
keywords = client.extract_sync("Apple announced new iPhone")
```

IMPLEMENTATION:
- Retry logic (3 attempts with exponential backoff)
- Proper error messages
- Type hints throughout
- ~200 lines

EXPECTED OUTPUT: Complete client.py that's easy to use
PROMPT 7/10: Requirements Minimal
markdownFOCUS FILE: requirements.minimal.txt

CONTEXT: Minimal dependencies for Fas 1. Keep it lean but functional.

REQUIREMENTS:
Core dependencies only:
```
# Core ML
sentence-transformers==2.2.2
spacy==3.5.3
torch>=2.0.0,<2.1.0  # CPU version for minimal

# NLP
networkx==3.1

# API
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Client
httpx==0.25.2

# Utilities
numpy==1.24.4
python-multipart==0.0.6  # For file uploads
beautifulsoup4==4.12.2  # For URL content extraction

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
```

NOTES:
- Pin versions for reproducibility
- Add comments for each section
- Include command to download spacy model

EXPECTED OUTPUT: Complete requirements.minimal.txt
PROMPT 8/10: Docker Setup
markdownFOCUS FILE: docker-compose.minimal.yml AND Dockerfile.minimal

CONTEXT: Docker setup for easy local running. Should work with one command.

REQUIREMENTS for docker-compose.minimal.yml:
```yaml
version: '3.8'

services:
  sie-x-api:
    build:
      context: .
      dockerfile: Dockerfile.minimal
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models  # For model caching
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

REQUIREMENTS for Dockerfile.minimal:
```dockerfile
FROM python:3.10-slim

# Install system deps
# Copy requirements
# Install Python deps
# Download spacy model
# Copy app code
# Run uvicorn

WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "sie_x.api.minimal_server:app", "--host", "0.0.0.0"]
```

EXPECTED OUTPUT: Both files complete and working
PROMPT 9/10: Demo Script
markdownFOCUS FILE: demo/quickstart.py

CONTEXT: Quick demo showing SIE-X value. Should work immediately after docker-compose up.

REQUIREMENTS:
```python
#!/usr/bin/env python3
"""
SIE-X Quick Start Demo
Shows keyword extraction capabilities
"""

import asyncio
import json
from sie_x.sdk.python.client import SIEXClient

# Demo texts covering different domains
DEMO_TEXTS = {
    "tech_news": """
    Apple Inc. announced its latest MacBook Pro featuring the new M3 chip. 
    The Cupertino-based company claims 30% better performance than the previous 
    generation. CEO Tim Cook highlighted the improved battery life and AI capabilities.
    """,
    
    "scientific": """
    Researchers at MIT have developed a new CRISPR gene-editing technique that 
    reduces off-target effects by 95%. The breakthrough could accelerate the 
    development of gene therapies for inherited diseases.
    """,
    
    "business": """
    Amazon's Q4 earnings exceeded analyst expectations with $170 billion in revenue. 
    The e-commerce giant saw significant growth in AWS cloud services and announced 
    plans to expand its logistics network in Europe.
    """
}

async def demo_basic_extraction():
    """Show basic keyword extraction"""
    print("🚀 SIE-X Demo - Basic Extraction\n")
    
    async with SIEXClient() as client:
        # Check health
        if not await client.health_check():
            print("❌ API is not running. Please run: docker-compose -f docker-compose.minimal.yml up")
            return
            
        # Extract from each demo text
        for domain, text in DEMO_TEXTS.items():
            print(f"\n📄 Analyzing {domain} text...")
            print("-" * 50)
            
            keywords = await client.extract(text, top_k=5)
            
            print("🔑 Top Keywords:")
            for kw in keywords:
                print(f"  • {kw['text']:<20} (score: {kw['score']:.3f}, type: {kw['type']})")

async def demo_batch_processing():
    """Show batch processing capabilities"""
    print("\n\n🚀 SIE-X Demo - Batch Processing\n")
    
    async with SIEXClient() as client:
        texts = list(DEMO_TEXTS.values())
        results = await client.extract_batch(texts, top_k=3)
        
        print(f"📊 Processed {len(texts)} documents in batch")
        print(f"🔑 Keywords per document: {[len(r) for r in results]}")

async def demo_url_analysis():
    """Show URL analysis (if implemented)"""
    # Optional for Fas 1

def main():
    print("=" * 60)
    print("SIE-X - Semantic Intelligence Engine X")
    print("Quick Start Demo")
    print("=" * 60)
    
    # Run demos
    asyncio.run(demo_basic_extraction())
    asyncio.run(demo_batch_processing())
    
    print("\n✅ Demo complete!")
    print("\n💡 Try the API directly:")
    print("   curl -X POST http://localhost:8000/extract \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"text\": \"Your text here\"}'")

if __name__ == "__main__":
    main()
```

EXPECTED OUTPUT: Runnable demo that shows SIE-X capabilities clearly
PROMPT 10/10: Simple Frontend Demo
markdownFOCUS FILE: demo/index.html

CONTEXT: Simple HTML+JS frontend for visual demo. No build process needed.

REQUIREMENTS:
```html



    SIE-X Demo
    
    


    
        SIE-X Keyword Extraction
        
        
        
            
            
                Extract Keywords
                Processing...
            
        
        
        
        
            Extracted Keywords
            
                
                    
                        
                        
                            Score: 
                            Type: 
                        
                    
                
            
        
    
    
    
        function sieXDemo() {
            return {
                inputText: '',
                keywords: [],
                loading: false,
                
                async extractKeywords() {
                    this.loading = true;
                    try {
                        const response = await fetch('http://localhost:8000/extract', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({text: this.inputText})
                        });
                        const data = await response.json();
                        this.keywords = data.keywords;
                    } catch (error) {
                        alert('Error: ' + error.message);
                    }
                    this.loading = false;
                }
            }
        }
    


```

EXPECTED OUTPUT: Simple but functional frontend, no build needed
2. SEO TRANSFORMER DESIGN - BACOWR INTEGRATION
python# sie_x/transformers/seo_transformer.py

class SEOTransformer:
    """
    Transformerar SIE-X till SEO/Backlink Intelligence Engine
    Designad för integration med BACOWR pipeline
    """
    
    def __init__(self, serp_analyzer=None, link_scorer=None):
        self.serp_analyzer = serp_analyzer
        self.link_scorer = link_scorer
        
    def transform_extraction(self, original_extract_func):
        """Wrapper som lägger till SEO-intelligens"""
        
        async def seo_extract(
            text: str,
            url: str = None,
            mode: str = "analyze",  # analyze, publisher, target
            serp_context: Dict = None,
            **kwargs
        ) -> Dict[str, Any]:
            
            # Kör original keyword extraction
            keywords = await original_extract_func(text, **kwargs)
            
            # SEO-specifik analys baserat på mode
            if mode == "publisher":
                return await self._analyze_publisher(text, keywords, url)
            elif mode == "target":
                return await self._analyze_target(text, keywords, url, serp_context)
            else:
                return await self._general_analysis(text, keywords, url)
    
    async def _analyze_publisher(self, text: str, keywords: List[Keyword], url: str) -> Dict:
        """Analysera publisher-sida för BACOWR"""
        return {
            # För BacklinkJobPackage.source_analysis
            "entities": self._extract_entities(keywords),
            "topics": self._cluster_topics(keywords),
            "content_type": self._classify_content_type(keywords, text),
            "authority_signals": self._extract_authority_signals(keywords),
            "link_placement_spots": self._find_link_spots(text, keywords),
            
            # För bridge-finding
            "semantic_themes": self._extract_themes(keywords),
            "concept_graph": self._build_concept_graph(keywords),
            
            # Original keywords
            "keywords": keywords
        }
    
    async def _analyze_target(self, text: str, keywords: List[Keyword], url: str, serp_context: Dict) -> Dict:
        """Analysera target-sida för BACOWR"""
        return {
            # För BacklinkJobPackage.target_analysis  
            "primary_entities": self._extract_primary_entities(keywords),
            "target_topics": self._identify_target_topics(keywords, serp_context),
            "content_gaps": self._find_content_gaps(keywords, serp_context),
            "serp_alignment": self._calculate_serp_alignment(keywords, serp_context),
            
            # För anchor text generation
            "anchor_candidates": self._generate_anchor_candidates(keywords),
            "anchor_risk_scores": self._score_anchor_risk(keywords),
            
            # För content brief
            "key_concepts": self._extract_key_concepts(keywords),
            "semantic_clusters": self._create_semantic_clusters(keywords)
        }
    
    def find_bridge_topics(self, publisher_analysis: Dict, target_analysis: Dict) -> List[Dict]:
        """Hitta semantiska bryggor mellan publisher och target"""
        bridges = []
        
        # Jämför topic-kluster
        publisher_topics = publisher_analysis['topics']
        target_topics = target_analysis['target_topics']
        
        for p_topic in publisher_topics:
            for t_topic in target_topics:
                similarity = self._calculate_topic_similarity(p_topic, t_topic)
                if similarity > 0.3:  # Threshold
                    bridges.append({
                        'publisher_topic': p_topic,
                        'target_topic': t_topic,
                        'similarity': similarity,
                        'bridge_type': self._classify_bridge_type(p_topic, t_topic),
                        'content_angle': self._suggest_angle(p_topic, t_topic)
                    })
        
        return sorted(bridges, key=lambda x: x['similarity'], reverse=True)
    
    def generate_content_brief(self, bridge: Dict, publisher_analysis: Dict, target_analysis: Dict) -> Dict:
        """Generera content brief för BACOWR writer"""
        return {
            # Integreras i generation_constraints
            'primary_topic': bridge['content_angle'],
            'must_mention_entities': target_analysis['primary_entities'][:3],
            'context_entities': publisher_analysis['entities'][:5],
            'semantic_keywords': self._merge_keywords(
                publisher_analysis['keywords'][:10],
                target_analysis['keywords'][:10]
            ),
            'content_gaps_to_fill': target_analysis['content_gaps'][:3],
            'tone_alignment': self._analyze_tone_match(publisher_analysis, target_analysis)
        }
3. BACOWR ADAPTER
python# sie_x/integrations/bacowr_adapter.py

class BACOWRAdapter:
    """Adapter för att integrera SIE-X i BACOWR pipeline"""
    
    def __init__(self, sie_x_client: SIEXClient):
        self.sie_x = sie_x_client
        self.seo_mode = True  # Aktivera SEO transformer
        
    async def enhance_page_profiler(self, profile: PageProfile) -> PageProfile:
        """Förbättra PageProfiler med SIE-X analys"""
        
        # Kör SIE-X analys
        analysis = await self.sie_x.extract(
            profile.text_content,
            url=profile.url,
            mode="publisher" if profile.is_publisher else "target"
        )
        
        # Berika profilen
        profile.semantic_entities = analysis['entities']
        profile.topic_clusters = analysis['topics']
        profile.content_type = analysis.get('content_type', 'unknown')
        
        return profile
    
    async def enhance_intent_analysis(self, intent_data: Dict, sie_x_analysis: Dict) -> Dict:
        """Förbättra intent-analys med semantic insights"""
        
        intent_data['semantic_alignment'] = self._calculate_semantic_alignment(
            intent_data['keywords'],
            sie_x_analysis['keywords']
        )
        
        intent_data['topic_relevance'] = sie_x_analysis.get('serp_alignment', {})
        
        return intent_data
    
    async def generate_smart_constraints(self, job: BacklinkJob, bridge: Dict) -> Dict:
        """Generera generation_constraints med SIE-X"""
        
        return {
            'content_requirements': {
                'primary_topic': bridge['content_angle'],
                'semantic_keywords': bridge['semantic_keywords'],
                'entities_to_mention': bridge['must_mention_entities'],
                'tone': bridge['tone_alignment']
            },
            'link_requirements': {
                'anchor_text_options': bridge['anchor_candidates'],
                'placement_context': bridge['optimal_placement_context'],
                'risk_score': bridge['anchor_risk_scores']
            },
            'quality_criteria': {
                'topic_relevance_min': 0.7,
                'semantic_coherence_min': 0.8,
                'entity_coverage_min': 0.6
            }
        }
4. QUICK WIN DEMO SETUP
bash# Steg-för-steg för Quick Win Demo

# 1. Klona repo
git clone https://github.com/robwestz/SEI-X.git
cd SEI-X

# 2. Skapa Fas 1 branch
git checkout -b fas-1-minimal

# 3. Kör Sonnet-prompts i Claude Code
# Kopiera varje prompt (1-10) och kör i Claude Code
# Spara filerna i rätt mappar

# 4. Starta med Docker
docker-compose -f docker-compose.minimal.yml up

# 5. Kör demo
python demo/quickstart.py

# 6. Öppna frontend
open demo/index.html
5. FAS 1 EXECUTION CHECKLIST
markdown## Fas 1 Implementation Checklist

### Core Files (Kör i Claude Code)
- [ ] sie_x/core/models.py (Prompt 1)
- [ ] sie_x/core/simple_engine.py (Prompt 2)
- [ ] sie_x/core/extractors.py (Prompt 3)

### API Files
- [ ] sie_x/api/minimal_server.py (Prompt 4)
- [ ] sie_x/api/routes.py (Prompt 5)

### SDK
- [ ] sie_x/sdk/python/client.py (Prompt 6)

### Infrastructure
- [ ] requirements.minimal.txt (Prompt 7)
- [ ] docker-compose.minimal.yml (Prompt 8)
- [ ] Dockerfile.minimal (Prompt 8)

### Demo
- [ ] demo/quickstart.py (Prompt 9)
- [ ] demo/index.html (Prompt 10)

### Test & Deploy
- [ ] Run: docker-compose -f docker-compose.minimal.yml up
- [ ] Test: python demo/quickstart.py
- [ ] Open: http://localhost:8000/docs
NÄSTA STEG

Kör alla 10 Sonnet-prompts i Claude Code
Committa till fas-1-minimal branch
Testa att allt fungerar med Docker
Visa demo för potentiella användare

När Fas 1 är klar och fungerar, går vi vidare till:

Fas 2: SEO Transformer implementation
Fas 3: Full BACOWR integration

Vill du att jag:

Generera fler detaljerade prompts för Fas 2?
Skapa en sales pitch för SIE-X + BACOWR?
Designa API endpoints specifikt för BACOWR-integration?
RetryRSka jag skicka alla samtidigt, köra varsin agent, ha nån introprompt för vad dom ska göra med den dom får?Analyserade beroenden och optimerade körordning för maximal effektivitet.10sAnvändaren har Claude Code credits som räcker en dag till och vill maximera användningen. De frågar om bästa sättet att använda de 10 prompts jag just skapat.
Låt mig tänka på det mest effektiva sättet:

Alla samtidigt - Nej, det blir rörigt och svårt att hålla koll på dependencies
Varsin agent - Ja, det kan vara smart men vissa filer beror på andra
Introprompt - Definitivt, det ger context

Bästa strategin med Claude Code:

Gruppera prompts baserat på dependencies
Kör i rätt ordning
Ha en tydlig introprompt som förklarar projektet och vad som ska byggas
Maximera varje agents output

Dependencies:

models.py (ingen dependency)
extractors.py (ingen dependency)
simple_engine.py (behöver models.py)
minimal_server.py (behöver simple_engine.py och models.py)
routes.py (behöver models.py)
client.py (behöver models.py)
requirements.minimal.txt (ingen dependency)
docker-compose.minimal.yml + Dockerfile.minimal (behöver veta strukturen)
quickstart.py (behöver client.py)
index.html (standalone)

Smart gruppering för Claude Code sessions.OPTIMAL CLAUDE CODE STRATEGI FÖR MAXIMALT VÄRDE
GRUPPERING & KÖRORDNING
Kör i 4 Claude Code sessions med tydlig gruppering:
SESSION 1: Core Foundation (Kör först!)
markdownINTRO PROMPT FOR SESSION 1:
You're building SIE-X (Semantic Intelligence Engine X), a keyword extraction system. 
This session focuses on the CORE DATA MODELS and EXTRACTION UTILITIES.
Create these files in order:
1. sie_x/core/models.py - Pydantic models
2. sie_x/core/extractors.py - Extraction utilities
3. requirements.minimal.txt - Dependencies

These are foundational files that other components depend on.
Make them production-ready and well-documented.
Sedan klistra in:

Prompt 1 (models.py)
Prompt 3 (extractors.py)
Prompt 7 (requirements.minimal.txt)

SESSION 2: Engine Implementation
markdownINTRO PROMPT FOR SESSION 2:
You're implementing the HEART of SIE-X - the extraction engine.
You have models.py and extractors.py already created.
Create:
1. sie_x/core/simple_engine.py - The main engine using sentence-transformers + spaCy

This is the most important file. Make it robust, fast, and well-tested.
Include inline tests/examples in docstrings.
Sedan:

Prompt 2 (simple_engine.py)

SESSION 3: API & SDK
markdownINTRO PROMPT FOR SESSION 3:
You're building the API layer and Python SDK for SIE-X.
The engine (simple_engine.py) and models are already done.
Create:
1. sie_x/api/minimal_server.py - FastAPI server
2. sie_x/api/routes.py - Additional routes
3. sie_x/sdk/python/client.py - Python client library

Make the API production-ready with proper error handling.
The SDK should be intuitive and well-documented.
Sedan:

Prompt 4 (minimal_server.py)
Prompt 5 (routes.py)
Prompt 6 (client.py)

SESSION 4: Docker & Demo
markdownINTRO PROMPT FOR SESSION 4:
You're creating the deployment and demo layer for SIE-X.
All core code is done. Now make it easy to run and demonstrate.
Create:
1. docker-compose.minimal.yml - Docker Compose setup
2. Dockerfile.minimal - Docker image
3. demo/quickstart.py - CLI demo
4. demo/index.html - Web demo

Focus on making it work with ONE command: docker-compose up
```

Sedan:
- Prompt 8 (Docker files)
- Prompt 9 (quickstart.py)
- Prompt 10 (index.html)

### **PRO TIPS FÖR CLAUDE CODE**

1. **Be om fler filer** när du får bra output:
```
   "Great! Now create unit tests for this in sie_x/tests/test_simple_engine.py"
```

2. **Be om relaterade utilities**:
```
   "Also create a sie_x/core/utils.py with helper functions for text preprocessing"
```

3. **Be om exempel och docs**:
```
   "Create a examples/basic_usage.py showing real-world usage"
```

4. **Be om optimeringar**:
```
   "Optimize the simple_engine.py for better performance with batch processing"
MAXIMERA VARJE SESSION
Efter varje fil, följ upp med:
markdownThe [filename] looks great! Now please also create:

1. Unit tests for this module in tests/test_[module].py
2. A focused README in the same directory explaining this component
3. Any missing utility functions this might need
4. Configuration file if applicable (config/[module].yaml)

Make sure everything is production-ready with proper error handling.
BONUS-FILER ATT BE OM
När grundfilerna är klara, maximera krediter med:
markdown## Extra värdefulla filer:

1. "Create sie_x/core/advanced_engine.py with GPU support and advanced features"

2. "Create sie_x/transformers/seo_transformer.py implementing the SEO transformation"

3. "Create sie_x/monitoring/metrics.py for Prometheus metrics"

4. "Create tests/integration/test_full_pipeline.py"

5. "Create sie_x/cache/redis_cache.py for production caching"

6. "Create deployment/kubernetes/sie-x-deployment.yaml"

7. "Create sie_x/ml/active_learning.py for the feedback loop"

8. "Create docs/API.md with full API documentation"

9. "Create sie_x/benchmarks/performance.py"

10. "Create sie_x/integrations/langchain_adapter.py"
EXEKVERINGSORDNING
bash# 1. Starta med Session 1 (Foundation)
# Vänta tills alla filer är genererade

# 2. Kör Session 2 (Engine)
# Detta är den viktigaste - be om extra features här

# 3. Kör Session 3 (API & SDK)
# Be om extra endpoints och SDK features

# 4. Kör Session 4 (Docker & Demo)
# Be om fler exempel och demos

# 5. Om du har krediter kvar - be om bonus-filer!
SMART UPPFÖLJNINGSSTRATEGI
Efter varje fil, evaluera:

Är koden komplett? Om inte → "Please complete the implementation of [specific function]"
Saknas error handling? → "Add comprehensive error handling for all edge cases"
Behövs tester? → "Create thorough unit tests for this module"
Kan det optimeras? → "Optimize this for production use with 1000 requests/second"

Detta sätt maximerar dina Claude Code-krediter genom att:

Få mest värde per session
Bygga i rätt ordning
Få bonus-content
Säkerställa production-ready kod

Vill du att jag förbereder några specifika uppföljningsprompts för varje session?RetryRkan jag sätta nån på att uppgradera och felsöka i det befintliga repots filer?Formulerade strategi för att uppgradera befintliga repofiler istället för nya.Användaren undrar om de kan använda Claude Code för att uppgradera och felsöka de befintliga filerna i repot istället för att skapa nya. Detta är en smart strategi eftersom repot redan har en massa filer från vår tidigare konversation som troligen är halvfärdiga eller behöver förbättras.
Jag bör ge en strategi för hur man bäst använder Claude Code för att:

Analysera befintliga filer
Identifiera problem/luckor
Uppgradera dem till production-ready
Felsöka och fixa issues

Detta kan vara mer värdefullt än att skapa nya filer från scratch.JA! SMART STRATEGI - UPPGRADERA BEFINTLIGT REPO
Det är faktiskt en BÄTTRE approach än att börja om. Här är hur du maximerar värdet:
UPPGRADERINGS-STRATEGI FÖR BEFINTLIGA FILER
SESSION 1: REPO AUDIT & FIX
markdownINTRO PROMPT:
You have access to the SEI-X repository at https://github.com/robwestz/SEI-X
Please analyze the existing codebase and:

1. First, explore the repo structure and identify all Python files
2. Check which files are incomplete, have TODOs, or are just stubs
3. Create a priority list of files that need completion
4. Start upgrading the most critical files to make them production-ready

Focus on making the CORE ENGINE actually work first.
For each file you upgrade:
- Complete all placeholder implementations
- Add proper error handling
- Add logging
- Add docstrings
- Fix any imports or dependencies
- Make it actually runnable

Start with sie_x/core/engine.py and make it fully functional.
SESSION 2: FIX BROKEN DEPENDENCIES
markdownINTRO PROMPT:
The SEI-X repo has many interconnected files. Your job is to:

1. Fix all broken imports between modules
2. Ensure consistent interfaces between components
3. Complete any half-implemented classes/functions
4. Add missing __init__.py files
5. Fix the requirements.txt to include all actual dependencies
6. Create a working setup.py for pip installation

Start by analyzing how components should connect, then fix each connection point.
Make sure sie_x can be imported as a package.
SESSION 3: MAKE IT RUNNABLE
markdownINTRO PROMPT:
Now make SEI-X actually runnable. Focus on:

1. Complete the FastAPI implementation in sie_x/api/server.py
2. Fix any async/await issues
3. Implement all the endpoint handlers that are just stubs
4. Connect the API to the actual engine
5. Add proper startup/shutdown handlers
6. Create a simple run.py script that starts everything

Test command should be: python run.py
API should respond at: http://localhost:8000
SESSION 4: TESTING & VALIDATION
markdownINTRO PROMPT:
Add comprehensive testing to ensure SEI-X works:

1. Create tests/ directory with proper structure
2. Write unit tests for core components
3. Add integration tests for the API
4. Create test_data/ with sample texts
5. Add a pytest.ini configuration
6. Create a Makefile with common commands
7. Add GitHub Actions workflow for CI

All tests should pass with: pytest
SPECIFIKA FILER ATT PRIORITERA
markdownPRIORITY FIX LIST:

HIGH PRIORITY (Core functionality):
1. sie_x/core/engine.py - Make extract() actually work
2. sie_x/api/server.py - Complete all endpoints
3. sie_x/core/models.py - Ensure all models are complete
4. sie_x/cache/manager.py - Implement basic caching

MEDIUM PRIORITY (Features):
5. sie_x/chunking/chunker.py - Complete the chunking logic
6. sie_x/multilingual/engine.py - At least basic language detection
7. sie_x/monitoring/observability.py - Basic metrics

LOW PRIORITY (Nice to have):
8. transformers/*.py - Complete at least one transformer
9. sie_x/agents/autonomous.py - Basic agent implementation
SMART UPPGRADERINGSPROMPTS
För varje fil, använd detta mönster:
markdownPlease upgrade sie_x/core/engine.py:

Current state: The file has class structure but most methods are empty
Goal: Make it fully functional with:
- Complete implementation of extract_async()
- Working embedding generation
- Actual PageRank implementation
- Proper error handling
- Logging throughout
- At least 3 working extraction modes

Use simple but effective implementations. For example:
- Use sentence-transformers for embeddings (all-MiniLM-L6-v2)
- Use spaCy for NER (en_core_web_sm)
- Use networkx for graph operations

Make sure it can actually process text and return keywords.
Test with: "Apple Inc. announced new iPhone with AI features."
FELSÖKNINGSSTRATEGI
markdownDEBUG AND FIX SESSION:

1. Run pylint/flake8 on all Python files and fix issues
2. Check for circular imports and resolve them
3. Ensure all relative imports work correctly
4. Fix any Python 3.10+ syntax issues
5. Add proper exception handling everywhere
6. Ensure all async functions are properly awaited

For each issue found:
- Explain what was wrong
- Show the fix
- Verify it works
BONUS: SKAPA MIGRATION GUIDE
markdownAfter analyzing the repo, create MIGRATION_GUIDE.md:

1. Document what was broken/incomplete
2. List all changes made
3. Show before/after examples
4. Provide testing instructions
5. List any remaining TODOs

This helps track the upgrade progress.
EXEKVERINGSSTEG
bash# 1. Klona repot lokalt först
git clone https://github.com/robwestz/SEI-X.git
cd SEI-X

# 2. Skapa upgrade branch
git checkout -b upgrade-to-working

# 3. Kör Session 1-4 i Claude Code
# Låt den analysera och uppgradera

# 4. Efter varje session, testa:
python -m sie_x.core.engine  # Ska inte krascha
python run.py  # Ska starta API
pytest  # Ska köra tester

# 5. Commit förbättringarna
git add .
git commit -m "Upgrade core engine to working state"
```

### **MAXIMERA VÄRDE**

Be Claude Code att:

1. **Skapa test för varje fix**:
```
   "After fixing engine.py, create comprehensive tests in tests/test_engine.py"
```

2. **Dokumentera förändringar**:
```
   "Create CHANGELOG.md documenting all fixes and improvements"
```

3. **Lägg till exempel**:
```
   "Create examples/quickstart.py showing the now-working functionality"
```

4. **Optimera prestanda**:
```
   "Now optimize the engine for better performance"
Detta approach ger dig:

✅ Ett faktiskt fungerande system (inte bara struktur)
✅ Fixade buggar och dependencies
✅ Production-ready kod
✅ Dokumentation av vad som faktiskt fungerar

Vill du att jag förbereder mer specifika uppgraderingsprompts för de viktigaste filerna?RetryRkan jag köra igång dom andra sessionsen också som ska skapa dom nya filerna?Analyserade parallell körning av sessioner och potentiella konflikter.Användaren undrar om de kan köra både uppgraderingsessions (för befintliga filer) OCH skapa-nya-filer sessions samtidigt. Det är en smart fråga om hur man maximerar Claude Code credits.
Jag bör:

Förklara att JA, det går att köra parallellt
Men varna för potentiella konflikter
Ge en smart strategi för att undvika problem
Föreslå vilka sessions som kan köras parallellt utan konflikt
JA! KÖR PARALLELLA SESSIONS - SMART PARALLELISERING
STRATEGI FÖR PARALLELL KÖRNING
Du kan absolut köra flera sessions samtidigt! Här är den optimala uppdelningen:
GRUPP A: UPPGRADERA BEFINTLIGT (Kan köras parallellt)
Session A1: Fix Core Engine (sie_x/core/engine.py + related)
Session A2: Fix API Layer (sie_x/api/*.py)  
Session A3: Fix Models & Utils (sie_x/core/models.py, utils.py)
Session A4: Fix Infrastructure (Docker, requirements, setup.py)
GRUPP B: SKAPA NYTT (Kan köras parallellt)
Session B1: Create Simple Engine (sie_x/core/simple_engine.py) - NY ENKLARE VERSION
Session B2: Create Minimal API (sie_x/api/minimal_server.py) - PARALLEL API
Session B3: Create SDK (sie_x/sdk/python/client.py)
Session B4: Create Demo (demo/*.py, demo/*.html)
KONFLIKTHANTERING
För att undvika konflikter:
1. ANVÄND OLIKA FILNAMN
python# Befintlig (uppgraderas):
sie_x/core/engine.py          → Full implementation
sie_x/api/server.py           → Complete API

# Ny (skapas från scratch):  
sie_x/core/simple_engine.py   → Minimal version
sie_x/api/minimal_server.py   → Fas 1 API
2. BRANCH-STRATEGI
bash# Skapa separata branches
git checkout -b upgrade-existing  # För Grupp A
git checkout -b create-minimal    # För Grupp B

# Senare kan du merga smart
git checkout main
git merge upgrade-existing
git merge create-minimal
OPTIMAL PARALLEL SESSION SETUP
SESSION SET 1: Core Functionality (Kör dessa 4 samtidigt)
markdownTAB 1 - Upgrade Existing Engine:
"Fix and complete sie_x/core/engine.py with full implementation"

TAB 2 - Create Simple Engine:
"Create new sie_x/core/simple_engine.py as minimal working version"

TAB 3 - Fix Existing API:
"Complete sie_x/api/server.py with all endpoints"

TAB 4 - Create Minimal API:
"Create sie_x/api/minimal_server.py for quick start"
SESSION SET 2: Supporting Files (Nästa våg)
markdownTAB 5 - SDK Creation:
"Create sie_x/sdk/python/client.py"

TAB 6 - Fix Models:
"Complete sie_x/core/models.py"

TAB 7 - Create Tests:
"Create comprehensive tests/"

TAB 8 - Demo Creation:
"Create demo/ with examples"
SMART KOORDINERING
markdownCOORDINATION PROMPT (lägg till i varje session):

NOTE: This is part of parallel development. Other sessions are:
- Upgrading existing engine.py (full version)
- Creating simple_engine.py (minimal version)
- Fixing existing API (server.py)
- Creating minimal API (minimal_server.py)

Your file should be self-contained and not depend on the others being complete.
Use clear interfaces so they can be swapped later.
EXEKVERINGSPLAN
bash# Terminal 1: Skapa mapstruktur för nya filer
mkdir -p sie_x/core
mkdir -p sie_x/api  
mkdir -p sie_x/sdk/python
mkdir -p demo
mkdir -p tests

# Starta alla Claude Code sessions
# Kopiera prompts till respektive session
# Låt dem köra!
FÖRDELAR MED PARALLEL APPROACH

Dubbel output: Du får både en "fixed" version OCH en "minimal" version
Fallback: Om uppgraderingen blir komplex har du minimal version
A/B Testing: Kan jämföra prestanda mellan versionerna
Snabbare: 8 sessions parallellt = 8x snabbare!

POST-PARALLEL INTEGRATION
Efter alla sessions är klara:
python# sie_x/__init__.py
"""
SIE-X - Semantic Intelligence Engine X

Usage:
  from sie_x import Engine  # Full version
  from sie_x import SimpleEngine  # Minimal version
"""

try:
    from .core.engine import SemanticIntelligenceEngine as Engine
except ImportError:
    print("Full engine not available, using simple version")
    from .core.simple_engine import SimpleSemanticEngine as Engine

# Always available
from .core.simple_engine import SimpleSemanticEngine as SimpleEngine
MAXIMERA VARJE PARALLELL SESSION
I varje session, efter första filen:
markdownExcellent! Since other teams are working on related components, please also create:

1. Interface documentation (interfaces.md) showing how this component connects
2. Standalone test file that doesn't depend on other components  
3. Mock implementations for testing without dependencies
4. README specific to this component
