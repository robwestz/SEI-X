#!/usr/bin/env python
# SIE-X Demo Script

import asyncio
from sie_x.core.engine import SemanticIntelligenceEngine
from transformers.loader import TransformerLoader

async def main():
    print("🎯 SIE-X Demo")
    print("-" * 50)

    # Create base engine
    engine = SemanticIntelligenceEngine()

    # Demo 1: Basic extraction
    print("\n1. Basic Keyword Extraction:")
    text = "Apple Inc. announced record profits in Q3 2024, driven by strong iPhone sales."
    keywords = await engine.extract_async(text)
    for kw in keywords[:5]:
        print(f"  - {kw.text}: {kw.score:.3f}")

    # Demo 2: Legal transformation
    print("\n2. Legal AI Transformation:")
    from transformers.legal_transformer import LegalTransformer
    legal = LegalTransformer()
    legal.inject(engine)

    legal_text = "According to Article 5 GDPR, personal data must be processed lawfully."
    result = await engine.extract_async(legal_text)
    print(f"  Legal entities found: {len(result['legal_entities'])}")

    # Demo 3: Hybrid system
    print("\n3. Hybrid Financial-Legal System:")
    engine2 = SemanticIntelligenceEngine()
    loader = TransformerLoader(engine2)
    loader.create_hybrid_system(['financial', 'legal'])

    hybrid_text = "Insider trading violations under SEC Rule 10b-5 caused AAPL stock to drop 5%."
    result = await engine2.extract_async(hybrid_text)
    print(f"  Financial insights: {len(result['financial']['financial_entities']['companies'])}")
    print(f"  Legal compliance issues: {len(result['legal']['legal_entities'])}")

    print("\n✅ Demo complete!")

if __name__ == "__main__":
    asyncio.run(main())
