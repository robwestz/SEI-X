"""
SIE-X Backlink Strategy Cookbook

This script demonstrates the complete flow:
1. Analyze Publisher & Target
2. Find Semantic Bridge
3. Generate Writer Constraints
4. Prepare Prompt for LLM

Prerequisites:
    pip install -r requirements.minimal.txt
    python -m spacy download en_core_web_sm
"""

import asyncio
import json
import os
from sie_x.sdk.python.client import SIEXClient
from sie_x.integrations.bacowr_adapter import BACOWRAdapter

# --- MOCK DATA (In real app, fetch URL content) ---
PUBLISHER_TEXT = """
Digital transformation is reshaping how modern enterprises operate. 
Cloud infrastructure and automated workflows are key drivers of efficiency.
Companies are looking for ways to streamline their operations using SaaS tools.
"""
PUBLISHER_URL = "https://tech-blog.example.com/digital-transformation"

TARGET_TEXT = """
Our AI-powered CRM software helps sales teams close more deals.
Automate follow-ups, track leads, and integrate with your email.
Best CRM for small businesses in 2024.
"""
TARGET_URL = "https://my-saas.example.com/ai-crm"

async def run_workflow():
    print("🚀 Starting SIE-X Backlink Strategy Workflow...\n")

    # 1. Initialize Engine
    # Note: In a real app, you might connect to a running API server,
    # but here we use the client which can also wrap the local engine if configured.
    # For this standalone example, we assume we are importing the logic directly.
    
    # Direct import for standalone usage (no API server needed)
    from sie_x.transformers.seo_transformer import SEOTransformer
    from sie_x.core.simple_engine import SimpleSemanticEngine
    
    print("1️⃣  Initializing Semantic Engine & Transformers...")
    engine = SimpleSemanticEngine() # Loads spaCy models
    transformer = SEOTransformer()  # Loads Sentence Transformers
    
    # 2. Extract & Analyze
    print("2️⃣  Analyzing Content...")
    pub_keywords = engine.extract(PUBLISHER_TEXT, top_k=10)
    tgt_keywords = engine.extract(TARGET_TEXT, top_k=10)
    
    pub_analysis = await transformer.analyze_publisher(
        PUBLISHER_TEXT, pub_keywords, PUBLISHER_URL
    )
    tgt_analysis = await transformer.analyze_target(
        TARGET_TEXT, tgt_keywords, TARGET_URL
    )
    
    print(f"   Publisher Topic: {pub_analysis['topics'][0] if pub_analysis['topics'] else 'General'}")
    print(f"   Target Topic:    {tgt_analysis['target_topics'][0] if tgt_analysis['target_topics'] else 'General'}")

    # 3. Find Bridge
    print("\n3️⃣  Finding Semantic Bridges...")
    bridges = transformer.find_bridge_topics(pub_analysis, tgt_analysis)
    
    if not bridges:
        print("❌ No bridge found. Topics are too distant.")
        return

    best_bridge = bridges[0]
    print(f"   ✅ Best Bridge Found: '{best_bridge['content_angle']}'")
    print(f"   📊 Type: {best_bridge['bridge_type'].upper()} (Strength: {best_bridge['strength']:.2f})")
    
    # 4. Generate Writer Constraints (The "Brief")
    print("\n4️⃣  Generating Writer Brief...")
    # Helper to generate brief directly without Adapter overhead for this demo
    brief = transformer.generate_content_brief(best_bridge, pub_analysis, tgt_analysis)
    
    constraints = {
        "content_requirements": {
            "primary_topic": brief['primary_topic'],
            "semantic_keywords": [k['text'] for k in brief['semantic_keywords']],
            "must_mention_entities": [e['text'] for e in brief['must_mention_entities']],
            "tone": brief['tone_alignment']
        },
        "link_requirements": {
            "bridge_type": best_bridge['bridge_type'],
            "recommended_anchor": tgt_analysis['anchor_candidates'][0] if tgt_analysis['anchor_candidates'] else "click here",
            "context_notes": f"Connect {best_bridge['content_angle']} to {tgt_analysis['target_topics'][0]}"
        }
    }
    
    print(json.dumps(constraints, indent=2))
    
    print("\n✅ Workflow Complete. Feed the JSON above + 'sie_x/prompts/writer_prompt.md' to your LLM.")

if __name__ == "__main__":
    asyncio.run(run_workflow())
