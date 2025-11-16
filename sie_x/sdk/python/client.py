"""
Python client library for SIE-X API.

This module provides a simple, async-friendly client for interacting
with the SIE-X keyword extraction API.

Usage:
    # Async usage
    async with SIEXClient() as client:
        keywords = await client.extract("Apple announced new iPhone")
    
    # Sync usage
    client = SIEXClient()
    keywords = client.extract_sync("Apple announced new iPhone")
"""

import httpx
from typing import List, Dict, Optional, Union, Any
import asyncio
from urllib.parse import urljoin
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SIEXClient:
    """
    Python client for SIE-X API.
    
    This client provides both async and sync methods for keyword extraction.
    
    Attributes:
        base_url: Base URL of the SIE-X API
        timeout: Request timeout in seconds
        api_key: Optional API key for authentication (Phase 2+)
    
    Example:
        >>> async with SIEXClient("http://localhost:8000") as client:
        ...     keywords = await client.extract("Your text here")
        ...     for kw in keywords:
        ...         print(f"{kw['text']}: {kw['score']}")
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
        max_retries: int = 3
    ):
        """
        Initialize SIE-X client.
        
        Args:
            base_url: Base URL of the API (default: http://localhost:8000)
            timeout: Request timeout in seconds (default: 30.0)
            api_key: Optional API key for authentication
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.api_key = api_key
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers=self._get_headers()
        )
        return self
    
    async def __aexit__(self, *args):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        return urljoin(self.base_url + '/', endpoint.lstrip('/'))
    
    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method (get, post, etc.)
            endpoint: API endpoint
            **kwargs: Additional arguments for httpx
        
        Returns:
            HTTP response
        
        Raises:
            httpx.HTTPError: If all retries fail
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' or call extract_sync()")
        
        url = self._build_url(endpoint)
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = await self._client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All retry attempts failed: {e}")
        
        raise last_error
    
    async def extract(
        self,
        text: str,
        top_k: int = 10,
        min_confidence: float = 0.3,
        url: Optional[str] = None,
        include_entities: bool = True,
        include_concepts: bool = True,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            top_k: Maximum number of keywords to return
            min_confidence: Minimum confidence threshold (0-1)
            url: Optional source URL for metadata
            include_entities: Include named entities
            include_concepts: Include concept keywords
            language: Language code (default: "en")
        
        Returns:
            List of keyword dictionaries
        
        Example:
            >>> keywords = await client.extract("Machine learning is amazing!")
            >>> print(keywords[0])
            {'text': 'machine learning', 'score': 0.92, 'type': 'CONCEPT'}
        """
        payload = {
            "text": text,
            "url": url,
            "options": {
                "top_k": top_k,
                "min_confidence": min_confidence,
                "include_entities": include_entities,
                "include_concepts": include_concepts,
                "language": language
            }
        }
        
        response = await self._request_with_retry("POST", "/extract", json=payload)
        data = response.json()
        
        return data.get("keywords", [])
    
    async def extract_batch(
        self,
        texts: List[str],
        top_k: int = 10,
        min_confidence: float = 0.3,
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        Extract keywords from multiple texts in batch.
        
        Args:
            texts: List of texts to process
            top_k: Maximum keywords per text
            min_confidence: Minimum confidence threshold
            **kwargs: Additional options
        
        Returns:
            List of keyword lists (one per input text)
        
        Example:
            >>> texts = ["First doc", "Second doc"]
            >>> results = await client.extract_batch(texts)
            >>> len(results)
            2
        """
        payload = {
            "items": [{"text": text} for text in texts],
            "options": {
                "top_k": top_k,
                "min_confidence": min_confidence,
                **kwargs
            }
        }
        
        response = await self._request_with_retry("POST", "/extract/batch", json=payload)
        data = response.json()
        
        # Extract keywords from each response
        return [item.get("keywords", []) for item in data]
    
    async def analyze_url(
        self,
        url: str,
        top_k: int = 10,
        min_confidence: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Analyze a URL and extract keywords from its content.
        
        Args:
            url: URL to analyze
            top_k: Maximum number of keywords
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of keyword dictionaries
        """
        params = {
            "url": url,
            "top_k": top_k,
            "min_confidence": min_confidence
        }
        
        response = await self._request_with_retry(
            "POST",
            "/api/v1/analyze/url",
            params=params
        )
        data = response.json()
        
        return data.get("keywords", [])
    
    async def analyze_file(
        self,
        file_path: Union[str, Path],
        top_k: int = 10,
        min_confidence: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Analyze a file and extract keywords.
        
        Args:
            file_path: Path to file (txt, html, md)
            top_k: Maximum number of keywords
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of keyword dictionaries
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {"file": (file_path.name, f, "text/plain")}
            params = {
                "top_k": top_k,
                "min_confidence": min_confidence
            }
            
            response = await self._request_with_retry(
                "POST",
                "/api/v1/analyze/file",
                files=files,
                params=params
            )
        
        data = response.json()
        return data.get("keywords", [])
    
    async def health_check(self) -> bool:
        """
        Check if API is healthy.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            response = await self._request_with_retry("GET", "/health")
            data = response.json()
            return data.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get API statistics.
        
        Returns:
            Dictionary with API statistics
        """
        response = await self._request_with_retry("GET", "/stats")
        return response.json()
    
    async def get_models(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        response = await self._request_with_retry("GET", "/models")
        return response.json()
    
    # Synchronous wrappers for convenience
    
    def extract_sync(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Synchronous version of extract().
        
        Args:
            text: Text to extract keywords from
            **kwargs: Additional arguments for extract()
        
        Returns:
            List of keyword dictionaries
        """
        return asyncio.run(self._extract_with_client(text, **kwargs))
    
    async def _extract_with_client(self, text: str, **kwargs):
        """Helper for sync extract."""
        async with self:
            return await self.extract(text, **kwargs)
    
    def extract_batch_sync(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        Synchronous version of extract_batch().
        
        Args:
            texts: List of texts
            **kwargs: Additional arguments
        
        Returns:
            List of keyword lists
        """
        return asyncio.run(self._extract_batch_with_client(texts, **kwargs))
    
    async def _extract_batch_with_client(self, texts: List[str], **kwargs):
        """Helper for sync batch extract."""
        async with self:
            return await self.extract_batch(texts, **kwargs)
    
    def health_check_sync(self) -> bool:
        """
        Synchronous version of health_check().
        
        Returns:
            True if healthy
        """
        return asyncio.run(self._health_check_with_client())
    
    async def _health_check_with_client(self):
        """Helper for sync health check."""
        async with self:
            return await self.health_check()


# Convenience function for quick usage
def extract_keywords(
    text: str,
    base_url: str = "http://localhost:8000",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Quick synchronous keyword extraction.
    
    This is a convenience function for simple use cases.
    For production use, create a SIEXClient instance.
    
    Args:
        text: Text to analyze
        base_url: API base URL
        **kwargs: Additional options
    
    Returns:
        List of keywords
    
    Example:
        >>> from sie_x.sdk.python.client import extract_keywords
        >>> keywords = extract_keywords("Machine learning rocks!")
        >>> print(keywords)
    """
    client = SIEXClient(base_url=base_url)
    return client.extract_sync(text, **kwargs)
