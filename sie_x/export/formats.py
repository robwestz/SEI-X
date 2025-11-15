"""
Export functionality for various formats.
"""

import json
import networkx as nx
from typing import List, Dict, Any
import pandas as pd

from ..core.engine import Keyword


class ExportManager:
    """Handle exports to various formats."""

    @staticmethod
    def to_json(keywords: List[Keyword], metadata: Dict[str, Any] = None) -> str:
        """Export to JSON format."""
        data = {
            "keywords": [kw.to_dict() for kw in keywords],
            "metadata": metadata or {}
        }
        return json.dumps(data, indent=2)

    @staticmethod
    def to_csv(keywords: List[Keyword]) -> str:
        """Export to CSV format."""
        df = pd.DataFrame([
            {
                "text": kw.text,
                "score": kw.score,
                "type": kw.type,
                "count": kw.count,
                "confidence": kw.confidence,
                "cluster": kw.semantic_cluster,
                "related_terms": "; ".join(kw.related_terms)
            }
            for kw in keywords
        ])
        return df.to_csv(index=False)

    @staticmethod
    def to_graphml(keywords: List[Keyword], graph: nx.Graph) -> str:
        """Export semantic graph to GraphML format."""
        # Add node attributes
        for node in graph.nodes():
            keyword = graph.nodes[node]['keyword']
            graph.nodes[node].update({
                'label': keyword.text,
                'score': float(keyword.score),
                'type': keyword.type
            })

        return '\n'.join(nx.generate_graphml(graph))

    @staticmethod
    def to_embeddings(keywords: List[Keyword]) -> Dict[str, List[float]]:
        """Export keyword embeddings for vector databases."""
        embeddings = {}
        for kw in keywords:
            if kw.embeddings is not None:
                embeddings[kw.text] = kw.embeddings.tolist()
        return embeddings


```

### **4. PROJEKTSTRUKTUR**
```
sie_x /
├── __init__.py
├── core /
│   ├── __init__.py
│   ├── engine.py  # Main engine implementation
│   ├── models.py  # Data models and enums
│   └── utils.py  # Helper functions
├── chunking /
│   ├── __init__.py
│   ├── chunker.py  # Document chunking
│   └── strategies.py  # Chunking strategies
├── cache /
│   ├── __init__.py
│   ├── manager.py  # Cache management
│   └── backends.py  # Redis, Memory backends
├── graph /
│   ├── __init__.py
│   ├── builder.py  # Graph construction
│   └── optimizer.py  # Graph optimization algorithms
├── embeddings /
│   ├── __init__.py
│   ├── generator.py  # Embedding generation
│   └── index.py  # Vector index management
├── api /
│   ├── __init__.py
│   ├── server.py  # FastAPI server
│   ├── models.py  # API request/response models
│   └── middleware.py  # Auth, rate limiting
├── export /
│   ├── __init__.py
│   └── formats.py  # Export formatters
├── monitoring /
│   ├── __init__.py
│   ├── metrics.py  # Performance metrics
│   └── logging.py  # Structured logging
├── training /
│   ├── __init__.py
│   ├── finetune.py  # Model fine-tuning
│   └── active_learning.py  # Active learning loop
├── integrations /
│   ├── __init__.py
│   ├── gcp.py  # Google Cloud integration
│   ├── supabase.py  # Supabase integration
│   └── make.py  # Make.com webhooks
└── tests /
├── __init__.py
├── test_engine.py
├── test_api.py
└── benchmarks.py