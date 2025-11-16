"""
Minimal FastAPI server for SIE-X keyword extraction.

This is the Phase 1 minimal API server exposing the SimpleSemanticEngine
via HTTP endpoints.

Run with:
    uvicorn sie_x.api.minimal_server:app --reload
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import time
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from collections import defaultdict
import asyncio
import json

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SIE-X API",
    description="Semantic Intelligence Engine X - Keyword Extraction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
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

# Statistics
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
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    client_ip = request.client.host if request.client else "unknown"
    
    # Skip rate limiting for health check
    if request.url.path == "/health":
        return await call_next(request)
    
    if not check_rate_limit(client_ip):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"error": "Rate limit exceeded. Please try again later."}
        )
    
    return await call_next(request)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


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
            "docs": "/docs"
        }
    }


@app.post("/extract", response_model=ExtractionResponse, tags=["extraction"])
async def extract_keywords(request: ExtractionRequest):
    """
    Extract keywords from text.
    
    Args:
        request: ExtractionRequest with text and options
    
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
        
        # Update stats
        stats["total_extractions"] += 1
        stats["total_processing_time"] += processing_time
        
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
async def extract_batch(request: BatchExtractionRequest):
    """
    Extract keywords from multiple texts in batch.
    
    Args:
        request: BatchExtractionRequest with multiple items
    
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
async def extract_stream(request: ExtractionRequest):
    """
    Stream keyword extraction for large documents.

    Processes text in chunks and streams results using Server-Sent Events (SSE).
    Useful for documents >10K words or real-time UI updates.

    Args:
        request: ExtractionRequest with text and options

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
async def extract_multilang(request: ExtractionRequest):
    """
    Extract keywords with automatic language detection.

    Automatically detects the language of the input text and uses
    the appropriate spaCy model for that language.

    Supported languages: en, es, fr, de, it, pt, nl, el, nb, lt

    Args:
        request: ExtractionRequest with text and options
            - Add "language": "es" in metadata to force a specific language

    Returns:
        ExtractionResponse with keywords and detected language in metadata

    Example:
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
