"""
SIE-X Python SDK
Enterprise-grade SDK for Semantic Intelligence Engine X
"""

from typing import List, Dict, Any, Optional, Union, AsyncIterator
import asyncio
import httpx
import websockets
import json
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import backoff
from datetime import datetime, timedelta
import jwt
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class ExtractionMode(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    ADVANCED = "advanced"
    ULTRA = "ultra"


class OutputFormat(Enum):
    OBJECT = "object"
    STRING = "string"
    JSON = "json"


@dataclass
class Keyword:
    """Keyword extraction result."""
    text: str
    score: float
    type: str
    count: int
    related_terms: List[str]
    confidence: float
    semantic_cluster: Optional[int] = None


@dataclass
class ExtractionOptions:
    """Extraction configuration options."""
    top_k: int = 10
    mode: ExtractionMode = ExtractionMode.BALANCED
    enable_clustering: bool = True
    min_confidence: float = 0.3
    language: Optional[str] = None
    output_format: OutputFormat = OutputFormat.OBJECT


class SIEXAuth:
    """Authentication handler for SIE-X API."""

    def __init__(
            self,
            api_key: Optional[str] = None,
            jwt_token: Optional[str] = None,
            oauth_client_id: Optional[str] = None,
            oauth_client_secret: Optional[str] = None
    ):
        self.api_key = api_key
        self.jwt_token = jwt_token
        self.oauth_client_id = oauth_client_id
        self.oauth_client_secret = oauth_client_secret
        self._token_expiry = None

    async def get_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        if self.api_key:
            return {"Authorization": f"ApiKey {self.api_key}"}
        elif self.jwt_token:
            if self._is_token_expired():
                await self._refresh_token()
            return {"Authorization": f"Bearer {self.jwt_token}"}
        elif self.oauth_client_id:
            if not self.jwt_token or self._is_token_expired():
                await self._oauth_authenticate()
            return {"Authorization": f"Bearer {self.jwt_token}"}
        else:
            raise ValueError("No authentication method configured")

    def _is_token_expired(self) -> bool:
        """Check if JWT token is expired."""
        if not self._token_expiry:
            return True
        return datetime.now() >= self._token_expiry

    async def _oauth_authenticate(self):
        """Authenticate using OAuth2."""
        # Implement OAuth2 flow
        pass


class SIEXClient:
    """
    Main client for interacting with SIE-X API.

    Example:
```python
        from sie_x import SIEXClient, ExtractionOptions

        # Initialize client
        client = SIEXClient(api_key="your-api-key")

        # Extract keywords
        keywords = await client.extract(
            "Your text here",
            options=ExtractionOptions(top_k=20, mode=ExtractionMode.ADVANCED)
        )

        # Batch processing
        results = await client.extract_batch(
            ["doc1", "doc2", "doc3"],
            options=ExtractionOptions()
        )
```
    """

    def __init__(
            self,
            base_url: str = "https://api.sie-x.com",
            auth: Optional[SIEXAuth] = None,
            api_key: Optional[str] = None,
            timeout: float = 30.0,
            max_retries: int = 3,
            enable_caching: bool = True
    ):
        self.base_url = base_url.rstrip('/')
        self.auth = auth or SIEXAuth(api_key=api_key)
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_caching = enable_caching
        self._cache = {} if enable_caching else None
        self._session = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.aclose()

    @backoff.on_exception(
        backoff.expo,
        (httpx.HTTPError, httpx.TimeoutException),
        max_tries=3
    )
    async def _request(
            self,
            method: str,
            endpoint: str,
            **kwargs
    ) -> httpx.Response:
        """Make HTTP request with retry logic."""
        if not self._session:
            self._session = httpx.AsyncClient(timeout=self.timeout)

        headers = await self.auth.get_headers()
        headers.update(kwargs.get('headers', {}))
        kwargs['headers'] = headers

        url = f"{self.base_url}{endpoint}"
        response = await self._session.request(method, url, **kwargs)
        response.raise_for_status()

        return response

    async def extract(
            self,
            text: Union[str, List[str]],
            options: Optional[ExtractionOptions] = None
    ) -> Union[List[Keyword], List[str], Dict[str, Any]]:
        """
        Extract keywords from text(s).

        Args:
            text: Single text or list of texts
            options: Extraction configuration

        Returns:
            Extracted keywords in specified format
        """
        options = options or ExtractionOptions()

        # Check cache
        if self.enable_caching and isinstance(text, str):
            cache_key = self._get_cache_key(text, options)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for key: {cache_key}")
                return self._cache[cache_key]

        # Prepare request
        payload = {
            "text": text,
            "top_k": options.top_k,
            "mode": options.mode.value,
            "enable_clustering": options.enable_clustering,
            "min_confidence": options.min_confidence
        }

        if options.language:
            payload["language"] = options.language

        # Make request
        response = await self._request(
            "POST",
            "/extract",
            json=payload
        )

        result = response.json()

        # Parse response based on format
        if options.output_format == OutputFormat.OBJECT:
            keywords = [
                Keyword(**kw) for kw in result["keywords"]
            ]
            parsed_result = keywords
        elif options.output_format == OutputFormat.STRING:
            parsed_result = [kw["text"] for kw in result["keywords"]]
        else:
            parsed_result = result

        # Cache result
        if self.enable_caching and isinstance(text, str):
            self._cache[cache_key] = parsed_result

        return parsed_result

    async def extract_batch(
            self,
            documents: List[str],
            options: Optional[ExtractionOptions] = None,
            batch_size: int = 50
    ) -> List[List[Keyword]]:
        """
        Process multiple documents in batch.

        Args:
            documents: List of documents to process
            options: Extraction configuration
            batch_size: Number of documents per batch

        Returns:
            List of keyword lists for each document
        """
        results = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_results = await self.extract(batch, options)
            results.extend(batch_results)

        return results

    async def extract_stream(
            self,
            text: str,
            options: Optional[ExtractionOptions] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream extraction results as they're processed.

        Args:
            text: Text to process
            options: Extraction configuration

        Yields:
            Extraction results for each chunk
        """
        ws_url = self.base_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/extract/stream"

        headers = await self.auth.get_headers()

        async with websockets.connect(ws_url, extra_headers=headers) as websocket:
            # Send extraction request
            await websocket.send(json.dumps({
                "type": "extract",
                "text": text,
                "options": asdict(options) if options else {}
            }))

            # Receive results
            async for message in websocket:
                data = json.loads(message)
                if data["type"] == "complete":
                    break
                yield data

    async def analyze_multiple(
            self,
            documents: List[str],
            top_k_common: int = 10,
            top_k_distinctive: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze relationships across multiple documents.

        Args:
            documents: List of documents to analyze
            top_k_common: Number of common keywords to return
            top_k_distinctive: Number of distinctive keywords per document

        Returns:
            Analysis results with common and distinctive keywords
        """
        response = await self._request(
            "POST",
            "/analyze/multi",
            json={
                "documents": documents,
                "options": {
                    "top_k_common": top_k_common,
                    "top_k_distinctive": top_k_distinctive
                }
            }
        )

        return response.json()

    async def get_models(self) -> List[Dict[str, Any]]:
        """Get available models and their capabilities."""
        response = await self._request("GET", "/models")
        return response.json()

    async def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = await self._request("GET", "/health")
        return response.json()

    def _get_cache_key(self, text: str, options: ExtractionOptions) -> str:
        """Generate cache key for request."""
        key_data = f"{text}:{options.top_k}:{options.mode.value}:{options.min_confidence}"
        return hashlib.sha256(key_data.encode()).hexdigest()


class SIEXBatchProcessor:
    """
    High-performance batch processor for large-scale extraction.

    Example:
```python
        processor = SIEXBatchProcessor(client, concurrency=10)

        # Process files
        results = await processor.process_files(
            Path("documents/"),
            pattern="*.txt",
            options=ExtractionOptions()
        )
```
    """

    def __init__(
            self,
            client: SIEXClient,
            concurrency: int = 5,
            progress_callback: Optional[callable] = None
    ):
        self.client = client
        self.concurrency = concurrency
        self.progress_callback = progress_callback
        self._semaphore = asyncio.Semaphore(concurrency)

    async def process_files(
            self,
            directory: Path,
            pattern: str = "*",
            options: Optional[ExtractionOptions] = None
    ) -> Dict[str, List[Keyword]]:
        """Process all files in directory matching pattern."""
        files = list(directory.glob(pattern))
        results = {}

        async def process_file(file_path: Path):
            async with self._semaphore:
                try:
                    text = file_path.read_text(encoding='utf-8')
                    keywords = await self.client.extract(text, options)
                    results[str(file_path)] = keywords

                    if self.progress_callback:
                        self.progress_callback(file_path, keywords)

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    results[str(file_path)] = []

        tasks = [process_file(f) for f in files]
        await asyncio.gather(*tasks)

        return results


# Convenience functions
async def extract_keywords(
        text: str,
        api_key: str,
        **kwargs
) -> List[Keyword]:
    """Simple keyword extraction function."""
    async with SIEXClient(api_key=api_key) as client:
        return await client.extract(text, **kwargs)