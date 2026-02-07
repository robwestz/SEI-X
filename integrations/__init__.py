"""
SIE-X Integrations - Adapters for external systems.

This module provides adapters for integrating SIE-X with:
- BACOWR: Backlink Content Writer pipeline
- LangChain: LLM orchestration
- Other SEO/content tools
"""

from .bacowr_adapter import BACOWRAdapter

__all__ = ["BACOWRAdapter"]
