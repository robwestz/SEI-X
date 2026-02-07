"""
Minimal FastAPI server for SIE-X keyword extraction.

This is the Phase 1 minimal API server exposing the SimpleSemanticEngine
via HTTP endpoints.

Run with:
    uvicorn sie_x.api.minimal_server:app --reload
"""

from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
import time
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from collections import defaultdict
import asyncio
import json
from functools import lru_cache
from pathlib import Path

from sie_x.core.simple_engine import SimpleSemanticEngine
from sie_x.core.streaming import StreamingExtractor, ChunkConfig
from sie_x.core.multilang import MultiLangEngine
from sie_x.core.models import (
    ExtractionRequest,
    ExtractionResponse,
    BatchExtractionRequest,
    HealthResponse,
    Keyword,
    ExtractionOptions
)
from sie_x.api.auth import (
    Token,
    User,
    create_access_token,
    get_current_active_user,
    verify_password,
    FAKE_USERS_DB,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from sie_x.monitoring.metrics import (
    get_metrics_app,
    REQUESTS_TOTAL,
    REQUEST_LATENCY,
    ERRORS_TOTAL,
    KEYWORDS_EXTRACTED,
    ACTIVE_REQUESTS
)
from sie_x.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Config
_api_cfg = get_config().api

# Create FastAPI app
app = FastAPI(
    title=_api_cfg.title,
    description="Semantic Intelligence Engine X - Keyword Extraction API",
    version=_api_cfg.version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount Prometheus metrics
app.mount("/metrics", get_metrics_app())

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=_api_cfg.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instances
engine: Optional[SimpleSemanticEngine] = None
streaming_extractor: Optional[StreamingExtractor] = None
multilang_engine: Optional[MultiLangEngine] = None

# Simple rate limiting
rate_limit_store: Dict[str, List[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 1.0  # seconds

# Track startup time for uptime calculation
startup_time: Optional[datetime] = None

# Statistics (Legacy internal stats, now we use Prometheus too)
stats = {
    "total_extractions": 0,
    "total_processing_time": 0.0,
    "errors": 0
}


def check_rate_limit(client_ip: str) -> bool:
    """
    Simple in-memory rate limiting.
    
    Args:
        client_ip: Client IP address
    
    Returns:
        True if request is allowed, False if rate limit exceeded
    """
    now = time.time()
    
    # Clean old requests
    rate_limit_store[client_ip] = [
        req_time for req_time in rate_limit_store[client_ip]
        if now - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Check limit
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Add this request
    rate_limit_store[client_ip].append(now)
    return True


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Middleware for monitoring and rate limiting."""
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    # Rate limiting logic
    client_ip = request.client.host if request.client else "unknown"
    if request.url.path != "/health" and not check_rate_limit(client_ip):
        ACTIVE_REQUESTS.dec()
        REQUESTS_TOTAL.labels(method=request.method, endpoint=request.url.path, status="429").inc()
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"error": "Rate limit exceeded. Please try again later."}
        )

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Record metrics
        REQUESTS_TOTAL.labels(
            method=request.method, 
            endpoint=request.url.path, 
            status=str(response.status_code)
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=request.method, 
            endpoint=request.url.path
        ).observe(process_time)
        
        return response
    except Exception as e:
        ERRORS_TOTAL.labels(type=type(e).__name__).inc()
        raise
    finally:
        ACTIVE_REQUESTS.dec()


@app.on_event("startup")
async def startup_event():
    """Initialize the engine on startup."""
    global engine, streaming_extractor, multilang_engine, startup_time

    logger.info("Starting SIE-X API server...")
    startup_time = datetime.now()

    try:
        # Initialize the semantic engine
        logger.info("Loading SimpleSemanticEngine...")
        engine = SimpleSemanticEngine()
        logger.info("SimpleSemanticEngine loaded successfully")

        # Initialize streaming extractor
        logger.info("Initializing StreamingExtractor...")
        streaming_extractor = StreamingExtractor(engine=engine)
        logger.info("StreamingExtractor initialized successfully")

        # Initialize multi-language engine
        logger.info("Initializing MultiLangEngine...")
        multilang_engine = MultiLangEngine(auto_detect=True, cache_size=5)
        logger.info("MultiLangEngine initialized successfully")

        # Test the engine
        test_keywords = engine.extract("test initialization", top_k=1)
        logger.info(f"Engine test successful: {len(test_keywords)} keywords extracted")

    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down SIE-X API server...")
    
    if engine:
        engine.clear_cache()
        logger.info("Engine cache cleared")


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SIE-X API",
        "version": "1.0.0",
        "description": "Semantic Intelligence Engine X - Keyword Extraction API",
        "endpoints": {
            "extract": "/extract",
            "batch": "/extract/batch",
            "stream": "/extract/stream",
            "multilang": "/extract/multilang",
            "languages": "/languages",
            "health": "/health",
            "models": "/models",
            "stats": "/stats",
            "metrics": "/metrics",
            "maps_routing_pack": "/knowledge/maps-routing-pack",
            "maps_routing_pack_page": "/knowledge/maps-routing-pack/page",
            "docs": "/docs"
        }
    }


@app.post("/token", response_model=Token, tags=["auth"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint to get JWT access token.
    
    Use username/password from form data.
    Default credentials (dev):
    - admin / admin
    - user / user
    """
    user_dict = FAKE_USERS_DB.get(form_data.username)
    if not user_dict:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = user_dict
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


# ---- Local knowledge pack: Maps routing docs (offline) ----

_WORK_IN_PROGRESS_ROOT = Path(__file__).resolve().parents[2]
_MAPS_VAULT_ROOT = _WORK_IN_PROGRESS_ROOT / "maps-knowledge-vault"
_MAPS_ROUTING_PACK_PATH = (
    _WORK_IN_PROGRESS_ROOT
    / "siex-knowledge-fusion"
    / "artifacts"
    / "maps-routing-pack.json"
)


@lru_cache(maxsize=1)
def _load_maps_routing_pack() -> Dict[str, Any]:
    if not _MAPS_ROUTING_PACK_PATH.exists():
        raise FileNotFoundError(f"Maps routing pack not found: {_MAPS_ROUTING_PACK_PATH}")
    return json.loads(_MAPS_ROUTING_PACK_PATH.read_text(encoding="utf-8"))


def _filter_pack_items(
    items: List[Dict[str, Any]],
    q: Optional[str],
    api: Optional[str],
) -> List[Dict[str, Any]]:
    out = items
    if api:
        api_l = api.strip().lower()
        out = [it for it in out if str(it.get("api", "")).lower() == api_l]

    if q:
        q_l = q.strip().lower()
        out = [
            it
            for it in out
            if q_l in str(it.get("title", "")).lower()
            or q_l in str(it.get("url", "")).lower()
            or q_l in str(it.get("section", "")).lower()
        ]
    return out


@app.get("/knowledge/maps-routing-pack", tags=["knowledge"])
async def get_maps_routing_pack(
    q: Optional[str] = None,
    api: Optional[str] = None,
    limit: int = 25,
    offset: int = 0,
):
    """Return metadata + a filtered slice of the offline Maps routing pack.

    Query params:
    - q: substring match across title/url/section
    - api: filter on API key (e.g. 'routes-api', 'directions-api')
    - limit/offset: pagination
    """
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    try:
        pack = _load_maps_routing_pack()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    items = list(pack.get("items", []) or [])
    filtered = _filter_pack_items(items, q=q, api=api)

    total = len(filtered)
    sliced = filtered[offset : offset + limit]

    return {
        "pack": pack.get("pack", {}),
        "source": {
            "pack_path": str(_MAPS_ROUTING_PACK_PATH),
            "vault_root": str(_MAPS_VAULT_ROOT),
        },
        "query": {"q": q, "api": api, "limit": limit, "offset": offset},
        "total": total,
        "items": sliced,
    }


@app.get("/knowledge/maps-routing-pack/page", tags=["knowledge"])
async def get_maps_routing_pack_page(file: str, max_chars: int = 200_000):
    """Return the Markdown content for a vault page referenced by the pack.

    The `file` value should match an item['file'] (e.g. 'pages/routes-api/overview.md').
    """
    if not file:
        raise HTTPException(status_code=400, detail="file is required")
    if max_chars < 1 or max_chars > 2_000_000:
        raise HTTPException(status_code=400, detail="max_chars must be between 1 and 2000000")

    try:
        requested = (_MAPS_VAULT_ROOT / file).resolve()
        vault_root = _MAPS_VAULT_ROOT.resolve()
        if not requested.is_relative_to(vault_root):
            raise HTTPException(status_code=400, detail="file must be within the maps vault")
    except AttributeError:
        # Python < 3.9 fallback (should not happen here, but keep safe)
        requested = (_MAPS_VAULT_ROOT / file).resolve()
        vault_root = _MAPS_VAULT_ROOT.resolve()
        if str(requested).lower().startswith(str(vault_root).lower()) is False:
            raise HTTPException(status_code=400, detail="file must be within the maps vault")

    if not requested.exists() or not requested.is_file():
        raise HTTPException(status_code=404, detail=f"page not found: {file}")

    content = requested.read_text(encoding="utf-8")
    if len(content) > max_chars:
        content = content[:max_chars]

    return {
        "file": file,
        "path": str(requested),
        "truncated": len(content) >= max_chars,
        "content": content,
    }


@app.post("/extract", response_model=ExtractionResponse, tags=["extraction"])
async def extract_keywords(
    request: ExtractionRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Extract keywords from text.
    
    Args:
        request: ExtractionRequest with text and options
        current_user: Authenticated user (Required)
    
    Returns:
        ExtractionResponse with extracted keywords
    
    Raises:
        HTTPException: If extraction fails or text is invalid
    """
    if not engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized"
        )
    
    try:
        start_time = time.time()
        
        # Get options or use defaults
        options = request.options or ExtractionOptions()
        
        # Extract keywords
        keywords = engine.extract(
            text=request.text,
            top_k=options.top_k,
            min_confidence=options.min_confidence,
            include_entities=options.include_entities,
            include_concepts=options.include_concepts
        )
        
        processing_time = time.time() - start_time
        
        # Update stats and metrics
        stats["total_extractions"] += 1
        stats["total_processing_time"] += processing_time
        KEYWORDS_EXTRACTED.inc(len(keywords))
        
        # Build response
        response = ExtractionResponse(
            keywords=keywords,
            processing_time=processing_time,
            version="1.0.0",
            metadata={
                "text_length": len(request.text),
                "url": request.url,
                "options": options.model_dump()
            }
        )
        
        logger.info(
            f"Extracted {len(keywords)} keywords in {processing_time:.3f}s "
            f"(text length: {len(request.text)})"
        )
        
        return response
        
    except ValueError as e:
        stats["errors"] += 1
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        stats["errors"] += 1
        logger.error(f"Extraction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(e)}"
        )


@app.post("/extract/batch", response_model=List[ExtractionResponse], tags=["extraction"])
async def extract_batch(
    request: BatchExtractionRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Extract keywords from multiple texts in batch.
    
    Args:
        request: BatchExtractionRequest with multiple items
        current_user: Authenticated user (Required)
    
    Returns:
        List of ExtractionResponse objects
    """
    if not engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized"
        )
    
    try:
        start_time = time.time()
        
        # Process each item
        results = []
        total_keywords = 0
        
        for item in request.items:
            # Use item options or batch options or defaults
            options = item.options or request.options or ExtractionOptions()
            
            item_start = time.time()
            keywords = engine.extract(
                text=item.text,
                top_k=options.top_k,
                min_confidence=options.min_confidence,
                include_entities=options.include_entities,
                include_concepts=options.include_concepts
            )
            item_time = time.time() - item_start
            
            total_keywords += len(keywords)
            
            response = ExtractionResponse(
                keywords=keywords,
                processing_time=item_time,
                version="1.0.0",
                metadata={
                    "text_length": len(item.text),
                    "url": item.url
                }
            )
            results.append(response)
            
            stats["total_extractions"] += 1
            stats["total_processing_time"] += item_time
        
        KEYWORDS_EXTRACTED.inc(total_keywords)
        
        total_time = time.time() - start_time
        logger.info(
            f"Batch processed {len(request.items)} items in {total_time:.3f}s "
            f"(avg: {total_time/len(request.items):.3f}s per item)"
        )
        
        return results

    except Exception as e:
        stats["errors"] += 1
        logger.error(f"Batch extraction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch extraction failed: {str(e)}"
        )


@app.post("/extract/stream", tags=["extraction"])
async def extract_stream(
    request: ExtractionRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Stream keyword extraction for large documents.

    Processes text in chunks and streams results using Server-Sent Events (SSE).
    Useful for documents >10K words or real-time UI updates.

    Args:
        request: ExtractionRequest with text and options
        current_user: Authenticated user (Required)

    Returns:
        StreamingResponse with SSE events containing chunk results

    Example SSE event:
        data: {"chunk_id": 0, "keywords": [...], "progress": 50.0, ...}
    """
    if not streaming_extractor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Streaming extractor not initialized"
        )

    # Get options or use defaults
    options = request.options or ExtractionOptions()

    async def event_generator():
        """Generate Server-Sent Events for streaming."""
        try:
            async for chunk_result in streaming_extractor.extract_stream(
                text=request.text,
                top_k=options.top_k,
                min_confidence=options.min_confidence,
                merge_final=True
            ):
                # Format as SSE
                data = json.dumps(chunk_result)
                yield f"data: {data}\n\n"

                # Update stats for merged result
                if chunk_result.get('is_merged_result'):
                    stats["total_extractions"] += 1
                    if 'keywords' in chunk_result:
                        KEYWORDS_EXTRACTED.inc(len(chunk_result['keywords']))

            # Send completion event
            yield "event: complete\ndata: {}\n\n"

        except Exception as e:
            stats["errors"] += 1
            logger.error(f"Streaming error: {e}", exc_info=True)
            error_data = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.post("/extract/multilang", response_model=ExtractionResponse, tags=["extraction"])
async def extract_multilang(
    request: ExtractionRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Extract keywords with automatic language detection.

    Automatically detects the language of the input text and uses
    the appropriate spaCy model for that language.

    Supported languages: en, sv, es, fr, de, it, pt, nl, el, nb, lt

    Args:
        request: ExtractionRequest with text and options
            - Add "language": "sv" in metadata to force a specific language
        current_user: Authenticated user (Required)

    Returns:
        ExtractionResponse with keywords and detected language in metadata

    Examples:
        {"text": "Hej världen", "options": {"top_k": 5}}
        -> Detects Swedish, returns Swedish keywords

        {"text": "Hola mundo", "options": {"top_k": 5}}
        -> Detects Spanish, returns Spanish keywords
    """
    if not multilang_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Multi-language engine not initialized"
        )

    try:
        start_time = time.time()

        # Get options or use defaults
        options = request.options or ExtractionOptions()

        # Check if language is forced via metadata
        forced_language = None
        if request.metadata and 'language' in request.metadata:
            forced_language = request.metadata['language']

        # Extract keywords with language detection
        keywords = multilang_engine.extract(
            text=request.text,
            language=forced_language,
            top_k=options.top_k,
            min_confidence=options.min_confidence,
            include_entities=options.include_entities,
            include_concepts=options.include_concepts
        )

        processing_time = time.time() - start_time

        # Detect language for metadata (if not forced)
        detected_lang = forced_language or multilang_engine.detect_language(request.text)

        # Update stats
        stats["total_extractions"] += 1
        stats["total_processing_time"] += processing_time
        KEYWORDS_EXTRACTED.inc(len(keywords))

        # Build response
        response = ExtractionResponse(
            keywords=keywords,
            processing_time=processing_time,
            version="1.0.0",
            metadata={
                "text_length": len(request.text),
                "url": request.url,
                "detected_language": detected_lang,
                "language_forced": forced_language is not None,
                "options": options.model_dump()
            }
        )

        logger.info(
            f"Multilang extracted {len(keywords)} keywords in {processing_time:.3f}s "
            f"(language: {detected_lang}, text length: {len(request.text)})"
        )

        return response

    except ValueError as e:
        stats["errors"] += 1
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        stats["errors"] += 1
        logger.error(f"Multilang extraction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multilang extraction failed: {str(e)}"
        )


@app.get("/languages", tags=["monitoring"])
async def list_languages():
    """
    List supported languages and multi-language engine statistics.

    Returns:
        Dictionary with supported languages and usage stats
    """
    if not multilang_engine:
        return {
            "status": "not_initialized",
            "supported_languages": []
        }

    try:
        stats = multilang_engine.get_stats()

        return {
            "status": "ready",
            "supported_languages": stats['supported_languages'],
            "loaded_languages": stats['loaded_languages'],
            "auto_detect_enabled": stats['auto_detect_enabled'],
            "default_language": stats['default_language'],
            "statistics": {
                "total_extractions": stats['total_extractions'],
                "languages_detected": stats['languages_detected'],
                "cache_hit_rate": stats['cache_hit_rate']
            }
        }

    except Exception as e:
        logger.error(f"Error listing languages: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with system status
    """
    if not engine or not startup_time:
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            models_loaded=[],
            uptime=0.0
        )
    
    try:
        # Calculate uptime
        uptime = (datetime.now() - startup_time).total_seconds()
        
        # Get loaded models
        engine_stats = engine.get_stats()
        models = [
            engine_stats.get("model_name", "unknown"),
            engine_stats.get("spacy_model", "unknown")
        ]
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            models_loaded=models,
            uptime=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="degraded",
            version="1.0.0",
            models_loaded=[],
            uptime=0.0
        )


@app.get("/models", tags=["monitoring"])
async def list_models():
    """
    List available models and their status.
    
    Returns:
        Dictionary with model information
    """
    if not engine:
        return {
            "status": "not_initialized",
            "models": []
        }
    
    try:
        engine_stats = engine.get_stats()
        
        return {
            "status": "ready",
            "models": [
                {
                    "name": engine_stats.get("model_name", "unknown"),
                    "type": "sentence_transformer",
                    "status": "loaded"
                },
                {
                    "name": engine_stats.get("spacy_model", "unknown"),
                    "type": "spacy_nlp",
                    "status": "loaded"
                }
            ],
            "cache_size": engine_stats.get("cache_size", 0)
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/stats", tags=["monitoring"])
async def get_stats():
    """
    Get API usage statistics.
    
    Returns:
        Dictionary with statistics
    """
    avg_time = (
        stats["total_processing_time"] / stats["total_extractions"]
        if stats["total_extractions"] > 0
        else 0.0
    )
    
    engine_stats = engine.get_stats() if engine else {}
    
    return {
        "api_stats": {
            "total_extractions": stats["total_extractions"],
            "total_processing_time": stats["total_processing_time"],
            "average_processing_time": avg_time,
            "errors": stats["errors"]
        },
        "engine_stats": engine_stats,
        "uptime": (datetime.now() - startup_time).total_seconds() if startup_time else 0.0
    }


# Run with: uvicorn sie_x.api.minimal_server:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "sie_x.api.minimal_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )