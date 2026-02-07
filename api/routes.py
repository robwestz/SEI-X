"""
Additional API routes for SIE-X.

This module provides extra endpoints for URL analysis, file uploads,
and keyword search functionality.
"""

from fastapi import APIRouter, Query, UploadFile, File, HTTPException, status
from typing import List, Optional, Dict, Any
import httpx
from bs4 import BeautifulSoup
import logging
import asyncio

from sie_x.core.models import ExtractionResponse, Keyword, ExtractionOptions

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["extraction"])

# Store for search (in-memory for Phase 1)
keyword_store: Dict[str, List[Keyword]] = {}


@router.post("/analyze/url", response_model=ExtractionResponse)
async def analyze_url(
    url: str = Query(..., description="URL to analyze"),
    top_k: int = Query(10, ge=1, le=100, description="Number of keywords"),
    min_confidence: float = Query(0.3, ge=0.0, le=1.0, description="Minimum confidence")
):
    """
    Analyze a URL and extract keywords from its content.
    
    Args:
        url: URL to fetch and analyze
        top_k: Maximum number of keywords
        min_confidence: Minimum confidence threshold
    
    Returns:
        ExtractionResponse with keywords from URL content
    """
    try:
        # Fetch URL content
        text = await fetch_url_content(url)
        
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text content found at URL"
            )
        
        # Import engine here to avoid circular imports
        from sie_x.api.minimal_server import engine
        
        if not engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Engine not initialized"
            )
        
        # Extract keywords
        import time
        start_time = time.time()
        
        keywords = engine.extract(
            text=text,
            top_k=top_k,
            min_confidence=min_confidence
        )
        
        processing_time = time.time() - start_time
        
        response = ExtractionResponse(
            keywords=keywords,
            processing_time=processing_time,
            version="1.0.0",
            metadata={
                "url": url,
                "text_length": len(text),
                "source": "url_fetch"
            }
        )
        
        logger.info(f"Analyzed URL: {url}, extracted {len(keywords)} keywords")
        
        return response
        
    except httpx.HTTPError as e:
        logger.error(f"Error fetching URL {url}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch URL: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error analyzing URL: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/analyze/file", response_model=ExtractionResponse)
async def analyze_file(
    file: UploadFile = File(..., description="File to analyze (txt, html, md)"),
    top_k: int = Query(10, ge=1, le=100),
    min_confidence: float = Query(0.3, ge=0.0, le=1.0)
):
    """
    Analyze an uploaded file and extract keywords.
    
    Supported formats: .txt, .html, .md
    
    Args:
        file: Uploaded file
        top_k: Maximum number of keywords
        min_confidence: Minimum confidence threshold
    
    Returns:
        ExtractionResponse with keywords from file content
    """
    try:
        # Read file content
        content = await file.read()
        
        # Extract text based on file type
        text = extract_text_from_file(file.filename, content)
        
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text content found in file"
            )
        
        # Import engine
        from sie_x.api.minimal_server import engine
        
        if not engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Engine not initialized"
            )
        
        # Extract keywords
        import time
        start_time = time.time()
        
        keywords = engine.extract(
            text=text,
            top_k=top_k,
            min_confidence=min_confidence
        )
        
        processing_time = time.time() - start_time
        
        response = ExtractionResponse(
            keywords=keywords,
            processing_time=processing_time,
            version="1.0.0",
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "text_length": len(text),
                "source": "file_upload"
            }
        )
        
        logger.info(f"Analyzed file: {file.filename}, extracted {len(keywords)} keywords")
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File analysis failed: {str(e)}"
        )


@router.get("/keywords/search")
async def search_keywords(
    q: str = Query(..., description="Search query", min_length=1),
    limit: int = Query(10, ge=1, le=100, description="Maximum results")
):
    """
    Search in extracted keywords (Phase 1: returns mock data).
    
    In future phases, this will search a database of extracted keywords.
    
    Args:
        q: Search query
        limit: Maximum number of results
    
    Returns:
        List of matching keywords
    """
    # Phase 1: Return mock data
    # In Phase 2+: Implement actual search in database
    
    mock_results = [
        {
            "text": q,
            "score": 0.95,
            "type": "CONCEPT",
            "documents": 5,
            "last_seen": "2024-01-15"
        }
    ]
    
    return {
        "query": q,
        "results": mock_results[:limit],
        "total": len(mock_results),
        "note": "Phase 1: Mock data. Real search coming in Phase 2."
    }


@router.get("/stats")
async def get_detailed_stats():
    """
    Get detailed API statistics.
    
    Returns:
        Detailed statistics about API usage
    """
    from sie_x.api.minimal_server import stats, engine, startup_time
    from datetime import datetime
    
    avg_time = (
        stats["total_processing_time"] / stats["total_extractions"]
        if stats["total_extractions"] > 0
        else 0.0
    )
    
    engine_stats = engine.get_stats() if engine else {}
    uptime = (datetime.now() - startup_time).total_seconds() if startup_time else 0.0
    
    return {
        "api": {
            "uptime_seconds": uptime,
            "total_requests": stats["total_extractions"],
            "total_errors": stats["errors"],
            "success_rate": (
                (stats["total_extractions"] - stats["errors"]) / stats["total_extractions"]
                if stats["total_extractions"] > 0
                else 0.0
            )
        },
        "performance": {
            "total_processing_time": stats["total_processing_time"],
            "average_processing_time": avg_time,
            "requests_per_second": (
                stats["total_extractions"] / uptime
                if uptime > 0
                else 0.0
            )
        },
        "engine": engine_stats
    }


# Utility functions

async def fetch_url_content(url: str, timeout: float = 10.0) -> str:
    """
    Fetch content from URL and extract text.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
    
    Returns:
        Extracted text content
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get("content-type", "")
        
        if "text/html" in content_type:
            # Parse HTML and extract text
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        elif "text/plain" in content_type:
            return response.text
        else:
            # Try to get text anyway
            return response.text


def extract_text_from_file(filename: str, content: bytes) -> str:
    """
    Extract text from uploaded file.
    
    Args:
        filename: Name of the file
        content: File content as bytes
    
    Returns:
        Extracted text
    """
    # Get file extension
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if ext in ['txt', 'md', 'markdown']:
        # Plain text or markdown
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return content.decode('latin-1')
            except:
                raise ValueError("Unable to decode file content")
    
    elif ext in ['html', 'htm']:
        # HTML file
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        
        # Clean whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    else:
        raise ValueError(f"Unsupported file type: .{ext}")
