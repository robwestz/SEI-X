#!/usr/bin/env python3
"""
SIE-X Quick Start Demo

This demo shows the keyword extraction capabilities of SIE-X across
different domains and use cases.

Run with:
    python demo/quickstart.py

Make sure the API is running:
    docker-compose -f docker-compose.minimal.yml up
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sie_x.sdk.python.client import SIEXClient

# Demo texts covering different domains
DEMO_TEXTS = {
    "tech_news": """
    Apple Inc. announced its latest MacBook Pro featuring the new M3 chip. 
    The Cupertino-based company claims 30% better performance than the previous 
    generation. CEO Tim Cook highlighted the improved battery life and AI capabilities.
    The device includes advanced machine learning processors and neural engines.
    """,
    
    "scientific": """
    Researchers at MIT have developed a new CRISPR gene-editing technique that 
    reduces off-target effects by 95%. The breakthrough could accelerate the 
    development of gene therapies for inherited diseases like sickle cell anemia
    and muscular dystrophy. The team published their findings in Nature Biotechnology.
    """,
    
    "business": """
    Amazon's Q4 earnings exceeded analyst expectations with $170 billion in revenue. 
    The e-commerce giant saw significant growth in AWS cloud services and announced 
    plans to expand its logistics network in Europe. Jeff Bezos praised the team's
    innovation in artificial intelligence and machine learning applications.
    """,
    
    "sports": """
    LeBron James scored 40 points in the Lakers' victory over the Golden State Warriors
    at the Staples Center. The NBA superstar demonstrated exceptional basketball skills
    and leadership. Coach Frank Vogel praised the team's defensive strategy and
    Stephen Curry's performance for the Warriors.
    """,
    
    "politics": """
    President Biden announced new climate change initiatives during a White House
    press conference. The comprehensive plan includes renewable energy investments,
    carbon emission reductions, and partnerships with European Union leaders.
    Environmental groups praised the administration's commitment to sustainability.
    """
}


async def demo_basic_extraction():
    """Show basic keyword extraction across different domains."""
    print("🚀 SIE-X Demo - Basic Extraction\n")
    print("=" * 70)
    
    async with SIEXClient() as client:
        # Check health
        if not await client.health_check():
            print("❌ API is not running!")
            print("\n💡 Please start the API first:")
            print("   docker-compose -f docker-compose.minimal.yml up")
            return False
        
        print("✅ API is healthy and ready\n")
        
        # Extract from each demo text
        for domain, text in DEMO_TEXTS.items():
            print(f"\n📄 Analyzing {domain.replace('_', ' ').title()} Text")
            print("-" * 70)
            
            keywords = await client.extract(text, top_k=5)
            
            if keywords:
                print("🔑 Top Keywords:")
                for i, kw in enumerate(keywords, 1):
                    print(
                        f"  {i}. {kw['text']:<25} "
                        f"(score: {kw['score']:.3f}, "
                        f"type: {kw['type']}, "
                        f"count: {kw.get('count', 1)})"
                    )
            else:
                print("  ⚠️  No keywords extracted")
        
        return True


async def demo_batch_processing():
    """Show batch processing capabilities."""
    print("\n\n🚀 SIE-X Demo - Batch Processing\n")
    print("=" * 70)
    
    async with SIEXClient() as client:
        texts = list(DEMO_TEXTS.values())
        
        print(f"📊 Processing {len(texts)} documents in batch...")
        
        results = await client.extract_batch(texts, top_k=3)
        
        print(f"✅ Successfully processed {len(results)} documents\n")
        
        # Show summary
        print("📈 Batch Results Summary:")
        for i, (domain, keywords) in enumerate(zip(DEMO_TEXTS.keys(), results), 1):
            top_keyword = keywords[0]['text'] if keywords else "N/A"
            print(f"  {i}. {domain:15} → Top keyword: '{top_keyword}'")


async def demo_options_and_filtering():
    """Show different extraction options."""
    print("\n\n🚀 SIE-X Demo - Extraction Options\n")
    print("=" * 70)
    
    async with SIEXClient() as client:
        sample_text = DEMO_TEXTS["tech_news"]
        
        # High confidence only
        print("\n🎯 High Confidence Keywords (min_confidence=0.7):")
        print("-" * 70)
        keywords = await client.extract(
            sample_text,
            top_k=10,
            min_confidence=0.7
        )
        for kw in keywords:
            print(f"  • {kw['text']:<30} (score: {kw['score']:.3f})")
        
        # More keywords with lower threshold
        print("\n📋 More Keywords (min_confidence=0.3, top_k=15):")
        print("-" * 70)
        keywords = await client.extract(
            sample_text,
            top_k=15,
            min_confidence=0.3
        )
        print(f"  Found {len(keywords)} keywords")
        for kw in keywords[:5]:
            print(f"  • {kw['text']:<30} (score: {kw['score']:.3f})")
        print(f"  ... and {len(keywords) - 5} more")


async def demo_entity_types():
    """Show entity type distribution."""
    print("\n\n🚀 SIE-X Demo - Entity Types\n")
    print("=" * 70)
    
    async with SIEXClient() as client:
        sample_text = DEMO_TEXTS["business"]
        
        keywords = await client.extract(sample_text, top_k=20)
        
        # Group by type
        by_type = {}
        for kw in keywords:
            entity_type = kw.get('type', 'UNKNOWN')
            if entity_type not in by_type:
                by_type[entity_type] = []
            by_type[entity_type].append(kw['text'])
        
        print("📊 Keywords by Entity Type:")
        for entity_type, terms in sorted(by_type.items()):
            print(f"\n  {entity_type}:")
            for term in terms[:5]:  # Show first 5
                print(f"    - {term}")


async def demo_stats_and_models():
    """Show API statistics and model information."""
    print("\n\n🚀 SIE-X Demo - System Information\n")
    print("=" * 70)
    
    async with SIEXClient() as client:
        # Get models
        models = await client.get_models()
        print("\n🤖 Loaded Models:")
        for model in models.get('models', []):
            print(f"  • {model.get('name')} ({model.get('type')})")
        print(f"  Cache size: {models.get('cache_size', 0)} embeddings")
        
        # Get stats
        stats = await client.get_stats()
        api_stats = stats.get('api_stats', {})
        
        print("\n📊 API Statistics:")
        print(f"  Total extractions: {api_stats.get('total_extractions', 0)}")
        print(f"  Average time: {api_stats.get('average_processing_time', 0):.3f}s")
        print(f"  Errors: {api_stats.get('errors', 0)}")


def print_header():
    """Print demo header."""
    print("\n")
    print("=" * 70)
    print(" " * 15 + "SIE-X - Semantic Intelligence Engine X")
    print(" " * 25 + "Quick Start Demo")
    print("=" * 70)
    print("\n")


def print_footer():
    """Print demo footer with next steps."""
    print("\n\n")
    print("=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)
    
    print("\n💡 Next Steps:\n")
    
    print("1. Try the API directly with curl:")
    print("   curl -X POST http://localhost:8000/extract \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"text\": \"Your text here\"}'")
    
    print("\n2. Explore the interactive API docs:")
    print("   http://localhost:8000/docs")
    
    print("\n3. Use the Python SDK in your code:")
    print("   from sie_x.sdk.python.client import SIEXClient")
    print("   async with SIEXClient() as client:")
    print("       keywords = await client.extract('Your text')")
    
    print("\n4. Try the web demo:")
    print("   open demo/index.html")
    
    print("\n")


async def main():
    """Run all demos."""
    print_header()
    
    try:
        # Run demos in sequence
        success = await demo_basic_extraction()
        
        if not success:
            return
        
        await demo_batch_processing()
        await demo_options_and_filtering()
        await demo_entity_types()
        await demo_stats_and_models()
        
        print_footer()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running demo: {e}")
        print("\nMake sure the API is running:")
        print("  docker-compose -f docker-compose.minimal.yml up")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
