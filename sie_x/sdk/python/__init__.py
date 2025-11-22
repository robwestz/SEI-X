"""
SIE-X Python SDK - Client library for SIE-X API.

Provides two SDK options:
- SIEXClient: Simple async client for basic keyword extraction
- Enterprise SDK: Advanced features with auth, batch processing, streaming
"""

# Simple client (recommended for most use cases)
from .client import SIEXClient

# Enterprise SDK classes
from .sie_x_sdk import (
    ExtractionMode,
    OutputFormat,
    Keyword,
    ExtractionOptions,
    SIEXAuth,
    SIEXClient as SIEXEnterpriseClient,  # Renamed to avoid conflict
    SIEXBatchProcessor,
)

__all__ = [
    # Simple client
    "SIEXClient",
    # Enterprise SDK
    "SIEXEnterpriseClient",
    "SIEXBatchProcessor",
    "SIEXAuth",
    "ExtractionMode",
    "OutputFormat",
    "Keyword",
    "ExtractionOptions",
]
