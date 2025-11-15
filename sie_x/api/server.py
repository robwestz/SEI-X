"""
FastAPI server for SIE-X engine exposure.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import json

from ..core.engine import SemanticIntelligenceEngine, ModelMode

app = FastAPI(
    title="Semantic Intelligence Engine X",
    version="3.0.0",
    description="Production-ready semantic keyword extraction and analysis"
)

# Global engine instance
engine = None


class ExtractionRequest(BaseModel):
    text: Union[str, List[str]] = Field(..., description="Text(s) to analyze")
    top_k: int = Field(10, description="Number of keywords to extract")
    mode: str = Field("balanced", description="Extraction mode")
    enable_clustering: bool = Field(True, description="Enable semantic clustering")
    min_confidence: float = Field(0.3, description="Minimum confidence threshold")


class BatchRequest(BaseModel):
    documents: List[str] = Field(..., description="List of documents to process")
    options: Dict[str, Any] = Field({}, description="Processing options")


@app.on_event("startup")
async def startup_event():
    """Initialize the engine on startup."""
    global engine
    engine = SemanticIntelligenceEngine(
        mode=ModelMode.BALANCED,
        enable_gpu=True,
        enable_monitoring=True
    )


@app.post("/extract")
async def extract_keywords(request: ExtractionRequest):
    """Extract keywords from text(s)."""
    try:
        results = await engine.extract_async(
            text=request.text,
            top_k=request.top_k,
            output_format='json',
            enable_clustering=request.enable_clustering,
            min_confidence=request.min_confidence
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/batch")
async def batch_extract(
        request: BatchRequest,
        background_tasks: BackgroundTasks
):
    """Process multiple documents in batch mode."""
    job_id = f"batch_{int(asyncio.get_event_loop().time())}"

    # Start background processing
    background_tasks.add_task(
        process_batch_job,
        job_id,
        request.documents,
        request.options
    )

    return {
        "job_id": job_id,
        "status": "processing",
        "document_count": len(request.documents)
    }


@app.get("/extract/stream")
async def stream_extraction(text: str):
    """Stream extraction results as they're processed."""

    async def generate():
        chunks = engine.chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            keywords = await engine.extract_async(chunk, top_k=5)
            yield json.dumps({
                "chunk": i,
                "keywords": [kw.to_dict() for kw in keywords]
            }) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )


@app.post("/analyze/multi")
async def analyze_multiple_documents(request: BatchRequest):
    """Analyze relationships across multiple documents."""
    results = await engine.extract_multiple_advanced(
        texts=request.documents,
        **request.options
    )
    return results


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine": "loaded" if engine else "not_loaded",
        "device": str(engine.device) if engine else "unknown"
    }


async def process_batch_job(job_id: str, documents: List[str], options: Dict):
    """Process batch job in background."""
    # Implementation would store results in cache/database
    for doc in documents:
        await engine.extract_async(doc, **options)