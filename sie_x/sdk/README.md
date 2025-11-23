# SIE-X Python SDK

**Official Python client library for SIE-X keyword extraction API**

The SDK provides both simple and enterprise-grade clients for interacting with SIE-X, supporting async/sync operations, batch processing, streaming, and advanced authentication.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Simple Client (SIEXClient)](#simple-client)
- [Enterprise Client (SIEXEnterpriseClient)](#enterprise-client)
- [Batch Processing](#batch-processing)
- [Streaming](#streaming)
- [Authentication](#authentication)
- [Configuration](#configuration)
- [Error Handling](#error-handling)
- [Examples](#examples)

---

## Overview

### What is the SDK?

The SIE-X SDK is a Python client library that provides programmatic access to the SIE-X API. It handles:

- HTTP communication with SIE-X API
- Authentication (API keys, JWT, OAuth2)
- Request/response serialization
- Error handling and retries
- Batch processing
- WebSocket streaming
- Async and sync interfaces

### SDK Structure

```
sie_x/sdk/
└── python/
    ├── client.py           # Simple async client (recommended)
    ├── sie_x_sdk.py        # Enterprise client with advanced features
    ├── __init__.py         # Exports both clients
    └── README.md          # This file
```

### Which Client to Use?

**Simple Client (`SIEXClient`)** - Recommended for most users
- ✅ Easy to use, minimal configuration
- ✅ Async/sync support
- ✅ Automatic retries
- ✅ Context manager support
- ⚠️ Basic auth only (API key)

**Enterprise Client (`SIEXEnterpriseClient`)** - For production applications
- ✅ OAuth2 authentication
- ✅ WebSocket streaming
- ✅ Batch processing with progress tracking
- ✅ File and URL analysis
- ✅ Advanced caching
- ⚠️ More complex configuration

---

## Installation

### From PyPI (when published)

```bash
pip install sie-x-sdk
```

### From Source

```bash
git clone https://github.com/robwestz/SEI-X.git
cd SEI-X
pip install -e .
```

### Dependencies

```bash
pip install httpx pydantic websockets backoff
```

---

## Quick Start

### 1. Start the API Server

```bash
uvicorn sie_x.api.minimal_server:app --port 8000
```

### 2. Use the Simple Client

```python
from sie_x.sdk.python import SIEXClient

# Create client
async with SIEXClient("http://localhost:8000") as client:
    # Extract keywords
    keywords = await client.extract("Apple announced new iPhone with AI")

    # Print results
    for kw in keywords:
        print(f"{kw['text']}: {kw['score']:.2f}")
```

**Output:**
```
Apple: 0.95
iPhone: 0.92
AI: 0.88
```

---

## Simple Client

### Basic Usage (Async)

```python
from sie_x.sdk.python import SIEXClient

async def main():
    # Initialize client
    client = SIEXClient(
        base_url="http://localhost:8000",
        timeout=30.0,
        max_retries=3
    )

    # Extract keywords
    text = "OpenAI releases GPT-4 with multimodal capabilities"
    keywords = await client.extract(text, top_k=10)

    # Process results
    for keyword in keywords:
        print(f"Keyword: {keyword['text']}")
        print(f"Score: {keyword['score']:.3f}")
        print(f"Type: {keyword['type']}")
        print(f"Confidence: {keyword['confidence']:.2f}")
        print()

    # Close client
    await client.close()

# Run
import asyncio
asyncio.run(main())
```

### Sync Usage

```python
from sie_x.sdk.python import SIEXClient

# Create client
client = SIEXClient("http://localhost:8000")

# Extract (sync)
keywords = client.extract_sync(
    text="Bitcoin price surges to new all-time high",
    top_k=5
)

print(keywords)
```

### Context Manager

```python
# Async context manager (recommended)
async with SIEXClient() as client:
    result = await client.extract("Your text here")

# Handles connection lifecycle automatically
```

### With API Key Authentication

```python
client = SIEXClient(
    base_url="https://api.sie-x.com",
    api_key="sk_live_abc123...",
    timeout=60.0
)

# API key automatically included in requests
keywords = await client.extract(text)
```

### Batch Extraction

```python
documents = [
    "First document about technology...",
    "Second document about business...",
    "Third document about science..."
]

# Extract from multiple documents
results = await client.batch_extract(
    texts=documents,
    top_k=10,
    min_confidence=0.5
)

# Process results
for i, keywords in enumerate(results):
    print(f"\nDocument {i+1}:")
    for kw in keywords[:5]:
        print(f"  - {kw['text']}: {kw['score']:.2f}")
```

### Health Check

```python
# Check API health
health = await client.health_check()

print(f"Status: {health['status']}")
print(f"Version: {health['version']}")
print(f"Uptime: {health['uptime_seconds']}s")
```

### Custom Options

```python
from sie_x.sdk.python import SIEXClient

keywords = await client.extract(
    text="Your text here",
    top_k=20,                    # Max keywords
    min_confidence=0.4,           # Confidence threshold
    language="en",                # Force language
    include_entities=True,        # Include NER
    include_phrases=True,         # Include phrases
    deduplicate=True             # Remove duplicates
)
```

---

## Enterprise Client

### Installation

```python
from sie_x.sdk.python import (
    SIEXEnterpriseClient,
    SIEXAuth,
    SIEXBatchProcessor,
    ExtractionMode,
    ExtractionOptions
)
```

### OAuth2 Authentication

```python
# Create auth handler
auth = SIEXAuth(
    client_id="your-client-id",
    client_secret="your-client-secret",
    auth_url="https://auth.sie-x.com/oauth/token"
)

# Initialize client
client = SIEXEnterpriseClient(
    base_url="https://api.sie-x.com",
    auth=auth
)

# Authenticate
await client.authenticate()

# Use client
keywords = await client.extract(text="Your text here")
```

### JWT Authentication

```python
auth = SIEXAuth(
    token="eyJ0eXAiOiJKV1QiLCJhbGc...",
    auth_type="jwt"
)

client = SIEXEnterpriseClient(base_url=API_URL, auth=auth)
```

### Extraction Modes

```python
from sie_x.sdk.python import ExtractionMode

# Fast mode (basic extraction)
keywords = await client.extract(
    text="Your text",
    mode=ExtractionMode.FAST
)

# Balanced mode (default)
keywords = await client.extract(
    text="Your text",
    mode=ExtractionMode.BALANCED
)

# Advanced mode (with NER, phrases, relationships)
keywords = await client.extract(
    text="Your text",
    mode=ExtractionMode.ADVANCED
)

# Ultra mode (all features + semantic clustering)
keywords = await client.extract(
    text="Your text",
    mode=ExtractionMode.ULTRA
)
```

### Batch Processor

```python
from sie_x.sdk.python import SIEXBatchProcessor

# Create processor
processor = SIEXBatchProcessor(
    client=client,
    max_concurrent=10,      # Parallel requests
    retry_failed=True       # Auto-retry failures
)

# Process large batch
documents = load_documents()  # 1000+ documents

results = await processor.process_batch(
    documents,
    top_k=10,
    show_progress=True      # Progress bar
)

# Results include success/failure tracking
print(f"Successful: {results['successful']}")
print(f"Failed: {results['failed']}")
print(f"Total time: {results['total_time']}s")
```

### Streaming Extraction

```python
# WebSocket streaming for large documents
long_document = load_large_file()  # 100K+ words

async for chunk_result in client.stream_extract(long_document):
    print(f"Chunk {chunk_result.chunk_id}")
    print(f"Progress: {chunk_result.progress}%")
    print(f"Keywords: {len(chunk_result.keywords)}")

    # Process chunk keywords
    for kw in chunk_result.keywords[:5]:
        print(f"  - {kw.text}: {kw.score:.2f}")
```

### File Analysis

```python
# Analyze file
with open("document.pdf", "rb") as f:
    keywords = await client.analyze_file(
        file=f,
        filename="document.pdf",
        top_k=20
    )
```

### URL Analysis

```python
# Analyze URL
keywords = await client.analyze_url(
    url="https://techcrunch.com/article",
    top_k=15,
    extract_metadata=True  # Get page title, description, etc.
)

# Results include page metadata
print(f"Title: {keywords['metadata']['title']}")
print(f"Description: {keywords['metadata']['description']}")
```

### Advanced Options

```python
from sie_x.sdk.python import ExtractionOptions

options = ExtractionOptions(
    top_k=20,
    min_confidence=0.5,
    language="auto",
    include_entities=True,
    include_phrases=True,
    include_related_terms=True,
    enable_clustering=True,
    cluster_threshold=0.7,
    deduplicate=True,
    normalize_scores=True
)

keywords = await client.extract(
    text="Your text",
    options=options
)
```

---

## Batch Processing

### Simple Batch

```python
# Process multiple texts
texts = ["Text 1", "Text 2", "Text 3"]

results = await client.batch_extract(texts, top_k=10)
```

### Concurrent Batch Processing

```python
import asyncio

async def process_document(client, text):
    return await client.extract(text, top_k=10)

# Process 100 documents concurrently
documents = load_documents(100)

tasks = [process_document(client, doc) for doc in documents]
results = await asyncio.gather(*tasks)
```

### Progress Tracking

```python
from tqdm import tqdm

results = []
for text in tqdm(documents, desc="Processing"):
    keywords = await client.extract(text)
    results.append(keywords)
```

---

## Streaming

### Stream Large Documents

```python
# For documents >10K words
large_doc = load_large_document()

# Stream extraction
async for result in client.stream_extract(large_doc, chunk_size=1000):
    if result['is_final']:
        print("Processing complete!")
        final_keywords = result['keywords']
    else:
        print(f"Progress: {result['progress']}%")
```

### Custom Chunk Strategy

```python
# Smart chunking (respects paragraphs)
async for result in client.stream_extract(
    text=large_doc,
    chunk_size=1000,
    strategy="smart"  # or "simple"
):
    process_chunk(result)
```

---

## Authentication

### API Key

```python
client = SIEXClient(
    base_url="https://api.sie-x.com",
    api_key="sk_live_abc123..."
)
```

### JWT Token

```python
client = SIEXEnterpriseClient(
    base_url="https://api.sie-x.com",
    auth=SIEXAuth(
        token="eyJ0eXAiOiJKV1QiLCJhbGc...",
        auth_type="jwt"
    )
)
```

### OAuth2

```python
auth = SIEXAuth(
    client_id="your-client-id",
    client_secret="your-client-secret",
    auth_url="https://auth.sie-x.com/oauth/token"
)

client = SIEXEnterpriseClient(base_url=API_URL, auth=auth)
await client.authenticate()
```

### Token Refresh

```python
# Automatic token refresh
client = SIEXEnterpriseClient(
    base_url=API_URL,
    auth=auth,
    auto_refresh=True  # Automatically refresh expired tokens
)
```

---

## Configuration

### Client Configuration

```python
from sie_x.sdk.python import SIEXClient

client = SIEXClient(
    base_url="http://localhost:8000",
    timeout=30.0,          # Request timeout (seconds)
    max_retries=3,         # Max retry attempts
    retry_backoff=2.0,     # Backoff multiplier
    verify_ssl=True,       # SSL verification
    api_key=None,          # API key (optional)
    headers={}             # Custom headers
)
```

### Environment Variables

```bash
# Set API URL
export SIE_X_API_URL="https://api.sie-x.com"

# Set API key
export SIE_X_API_KEY="sk_live_abc123..."

# Use in code
import os

client = SIEXClient(
    base_url=os.getenv("SIE_X_API_URL"),
    api_key=os.getenv("SIE_X_API_KEY")
)
```

---

## Error Handling

### Common Exceptions

```python
from httpx import HTTPStatusError, RequestError

try:
    keywords = await client.extract("Your text")
except HTTPStatusError as e:
    if e.response.status_code == 401:
        print("Authentication failed")
    elif e.response.status_code == 429:
        print("Rate limit exceeded")
    elif e.response.status_code == 503:
        print("Service unavailable")
except RequestError as e:
    print(f"Network error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Automatic Retries

```python
# Client automatically retries on transient failures
client = SIEXClient(
    base_url=API_URL,
    max_retries=3,         # Retry up to 3 times
    retry_backoff=2.0      # Exponential backoff (2s, 4s, 8s)
)

# Retries on:
# - Network errors
# - 5xx server errors
# - Timeouts
```

### Custom Error Handling

```python
import backoff

@backoff.on_exception(
    backoff.expo,
    HTTPStatusError,
    max_tries=5,
    giveup=lambda e: e.response.status_code < 500
)
async def extract_with_retry(text):
    return await client.extract(text)
```

---

## Examples

### Example 1: SEO Content Analysis

```python
from sie_x.sdk.python import SIEXClient
import asyncio

async def analyze_content(url):
    async with SIEXClient() as client:
        # Extract keywords from URL
        keywords = await client.analyze_url(url, top_k=20)

        # Group by type
        entities = [kw for kw in keywords if kw['type'] == 'ENTITY']
        phrases = [kw for kw in keywords if kw['type'] == 'PHRASE']

        print(f"\n=== SEO Analysis for {url} ===\n")

        print("Top Entities:")
        for kw in entities[:5]:
            print(f"  - {kw['text']}: {kw['score']:.2f}")

        print("\nKey Phrases:")
        for kw in phrases[:5]:
            print(f"  - {kw['text']}: {kw['score']:.2f}")

asyncio.run(analyze_content("https://example.com/article"))
```

### Example 2: Bulk Document Processing

```python
from sie_x.sdk.python import SIEXBatchProcessor, SIEXEnterpriseClient
import pandas as pd

async def process_corpus():
    # Load documents
    df = pd.read_csv("documents.csv")
    documents = df['text'].tolist()

    # Create processor
    client = SIEXEnterpriseClient(base_url=API_URL)
    processor = SIEXBatchProcessor(client, max_concurrent=10)

    # Process all documents
    results = await processor.process_batch(
        documents,
        top_k=15,
        show_progress=True
    )

    # Save results
    df['keywords'] = results['keywords']
    df.to_csv("processed_documents.csv", index=False)

    print(f"Processed {results['successful']} documents")
    print(f"Failed: {results['failed']}")

asyncio.run(process_corpus())
```

### Example 3: Real-time News Monitoring

```python
from sie_x.sdk.python import SIEXClient
import feedparser
import asyncio

async def monitor_rss_feed(feed_url):
    client = SIEXClient()

    while True:
        # Fetch RSS feed
        feed = feedparser.parse(feed_url)

        for entry in feed.entries:
            # Extract keywords from article
            keywords = await client.analyze_url(
                entry.link,
                top_k=10
            )

            # Check for target keywords
            target_found = any(
                kw['text'].lower() in ['ai', 'machine learning', 'gpt']
                for kw in keywords
            )

            if target_found:
                print(f"\n🔔 Relevant article found!")
                print(f"Title: {entry.title}")
                print(f"URL: {entry.link}")
                print(f"Keywords: {[kw['text'] for kw in keywords[:5]]}")

        # Wait 5 minutes
        await asyncio.sleep(300)

asyncio.run(monitor_rss_feed("https://techcrunch.com/feed/"))
```

### Example 4: Multi-language Content Analysis

```python
from sie_x.sdk.python import SIEXClient

async def analyze_multilingual():
    client = SIEXClient()

    texts = {
        'en': "Artificial intelligence is transforming healthcare",
        'es': "La inteligencia artificial está transformando la salud",
        'sv': "Artificiell intelligens förändrar sjukvården"
    }

    for lang, text in texts.items():
        keywords = await client.extract(
            text,
            language=lang,
            top_k=5
        )

        print(f"\n{lang.upper()}: {text}")
        print("Keywords:", [kw['text'] for kw in keywords])

asyncio.run(analyze_multilingual())
```

---

## Performance Tips

### 1. Connection Pooling

```python
# Reuse client instance
client = SIEXClient()

# Don't create new client for each request
for text in documents:
    keywords = await client.extract(text)  # ✅ Good

# Instead of:
for text in documents:
    async with SIEXClient() as client:      # ❌ Bad (creates new connection)
        keywords = await client.extract(text)
```

### 2. Batch Processing

```python
# Use batch endpoint for multiple documents
keywords_list = await client.batch_extract(documents)  # ✅ Good

# Instead of individual requests:
for doc in documents:
    keywords = await client.extract(doc)  # ❌ Slower
```

### 3. Streaming for Large Documents

```python
# Use streaming for documents >10K words
async for result in client.stream_extract(large_doc):  # ✅ Good
    process(result)

# Instead of single request:
keywords = await client.extract(large_doc)  # ❌ May timeout
```

### 4. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
async def extract_cached(text):
    return await client.extract(text)
```

---

## API Reference

See [API Documentation](../api/README.md) for full endpoint reference.

---

## Related Documentation

- [API Module](../api/README.md) - Server endpoints
- [Core Engine](../core/README.md) - Extraction engine
- [Transformers](../transformers/README.md) - Domain-specific extractors
- [Examples](../../examples/) - More code examples

---

## Support

- **GitHub Issues**: https://github.com/robwestz/SEI-X/issues
- **Documentation**: https://docs.sie-x.com
- **Email**: support@sie-x.com

---

**Last Updated:** 2025-11-23
**SDK Version:** 1.0.0
**Python Version:** 3.8+
