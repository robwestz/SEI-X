"""
SEO Transformer for SIE-X - Backlink Intelligence Layer

This transformer adds SEO-specific analysis on top of keyword extraction,
designed specifically for integration with BACOWR backlink pipeline.

Key capabilities:
- Publisher/Target analysis
- Bridge topic finding
- Anchor text generation with risk scoring
- Content gap identification
- SERP alignment analysis
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from sie_x.core.models import Keyword

logger = logging.getLogger(__name__)


class SEOTransformer:
    """
    Transforms SIE-X keyword extraction into SEO/backlink intelligence.
    
    Designed for integration with BACOWR (Backlink Content Writer) pipeline.
    Provides semantic analysis for:
    - Publisher profiling (source website analysis)
    - Target profiling (destination website analysis)
    - Bridge topic identification (semantic connections)
    - Anchor text generation with risk scoring
    - Content brief generation
    
    Example:
        >>> transformer = SEOTransformer()
        >>> publisher_analysis = await transformer.analyze_publisher(
        ...     text=publisher_content,
        ...     url="https://publisher.com/article"
        ... )
        >>> target_analysis = await transformer.analyze_target(
        ...     text=target_content,
        ...     url="https://target.com/landing-page",
        ...     serp_context={"query": "best laptops", "top_10": [...]}
        ... )
        >>> bridges = transformer.find_bridge_topics(
        ...     publisher_analysis,
        ...     target_analysis
        ... )
    """
    
    def __init__(
        self,
        min_topic_similarity: float = 0.3,
        max_anchor_length: int = 60,
        risk_threshold: float = 0.7
    ):
        """
        Initialize SEO Transformer.
        
        Args:
            min_topic_similarity: Minimum similarity for bridge topics (0-1)
            max_anchor_length: Maximum anchor text length in characters
            risk_threshold: Threshold for high-risk anchor detection (0-1)
        """
        self.min_topic_similarity = min_topic_similarity
        self.max_anchor_length = max_anchor_length
        self.risk_threshold = risk_threshold
        
        logger.info("SEOTransformer initialized")
    
    async def analyze_publisher(
        self,
        text: str,
        keywords: List[Keyword],
        url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze publisher (source) website for backlink opportunities.
        
        This analysis identifies:
        - Content themes and topics
        - Authority signals (expert entities, citations)
        - Potential link placement locations
        - Content type classification
        
        Args:
            text: Publisher page content
            keywords: Extracted keywords from SIE-X
            url: Publisher page URL
            metadata: Additional metadata (author, publish date, etc.)
        
        Returns:
            Dict with publisher analysis for BACOWR BacklinkJobPackage
        """
        logger.info(f"Analyzing publisher: {url}")
        
        # Extract entities by type
        entities = self._extract_entities(keywords)
        
        # Cluster into topics
        topics = self._cluster_topics(keywords)
        
        # Classify content type
        content_type = self._classify_content_type(keywords, text, metadata)
        
        # Find authority signals
        authority_signals = self._extract_authority_signals(keywords, text)
        
        # Identify link placement spots
        link_spots = self._find_link_placement_spots(text, keywords)
        
        # Extract semantic themes
        semantic_themes = self._extract_semantic_themes(keywords)
        
        # Build concept graph for bridge finding
        concept_graph = self._build_concept_graph(keywords)
        
        return {
            "url": url,
            "entities": entities,
            "topics": topics,
            "content_type": content_type,
            "authority_signals": authority_signals,
            "link_placement_spots": link_spots,
            "semantic_themes": semantic_themes,
            "concept_graph": concept_graph,
            "keywords": [kw.model_dump() for kw in keywords],
            "metadata": metadata or {}
        }
    
    async def analyze_target(
        self,
        text: str,
        keywords: List[Keyword],
        url: Optional[str] = None,
        serp_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze target (destination) website for backlink strategy.
        
        This analysis identifies:
        - Primary topics and entities
        - Content gaps vs. SERP competitors
        - SERP alignment score
        - Anchor text candidates
        - Key concepts for content briefs
        
        Args:
            text: Target page content
            keywords: Extracted keywords from SIE-X
            url: Target page URL
            serp_context: SERP data (query, top results, etc.)
        
        Returns:
            Dict with target analysis for BACOWR BacklinkJobPackage
        """
        logger.info(f"Analyzing target: {url}")
        
        # Extract primary entities (most important for target)
        primary_entities = self._extract_primary_entities(keywords)
        
        # Identify target topics
        target_topics = self._identify_target_topics(keywords, serp_context)
        
        # Find content gaps
        content_gaps = self._find_content_gaps(keywords, serp_context)
        
        # Calculate SERP alignment
        serp_alignment = self._calculate_serp_alignment(keywords, serp_context)
        
        # Generate anchor text candidates
        anchor_candidates = self._generate_anchor_candidates(keywords)
        
        # Score anchor risk
        anchor_risk_scores = self._score_anchor_risk(anchor_candidates)
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(keywords)
        
        # Create semantic clusters
        semantic_clusters = self._create_semantic_clusters(keywords)
        
        return {
            "url": url,
            "primary_entities": primary_entities,
            "target_topics": target_topics,
            "content_gaps": content_gaps,
            "serp_alignment": serp_alignment,
            "anchor_candidates": anchor_candidates,
            "anchor_risk_scores": anchor_risk_scores,
            "key_concepts": key_concepts,
            "semantic_clusters": semantic_clusters,
            "keywords": [kw.model_dump() for kw in keywords],
            "serp_context": serp_context or {}
        }
    
    def find_bridge_topics(
        self,
        publisher_analysis: Dict[str, Any],
        target_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find semantic bridges between publisher and target.
        
        Bridges are topic overlaps that can be used to naturally
        connect the publisher content to the target content.
        
        Args:
            publisher_analysis: Publisher analysis from analyze_publisher()
            target_analysis: Target analysis from analyze_target()
        
        Returns:
            List of bridge topics, sorted by similarity score
        """
        logger.info("Finding bridge topics")
        
        bridges = []
        
        publisher_topics = publisher_analysis.get('topics', [])
        target_topics = target_analysis.get('target_topics', [])
        
        # Compare each publisher topic with each target topic
        for p_topic in publisher_topics:
            for t_topic in target_topics:
                similarity = self._calculate_topic_similarity(p_topic, t_topic)
                
                if similarity > self.min_topic_similarity:
                    bridge_type = self._classify_bridge_type(p_topic, t_topic, similarity)
                    content_angle = self._suggest_content_angle(p_topic, t_topic)
                    
                    bridges.append({
                        'publisher_topic': p_topic,
                        'target_topic': t_topic,
                        'similarity': float(similarity),
                        'bridge_type': bridge_type,
                        'content_angle': content_angle,
                        'strength': self._calculate_bridge_strength(
                            p_topic, t_topic, similarity
                        )
                    })
        
        # Sort by strength (combination of similarity and other factors)
        bridges.sort(key=lambda x: x['strength'], reverse=True)
        
        logger.info(f"Found {len(bridges)} bridge topics")
        return bridges
    
    def generate_content_brief(
        self,
        bridge: Dict[str, Any],
        publisher_analysis: Dict[str, Any],
        target_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate content brief for BACOWR writer.
        
        Creates detailed instructions for content creation based on
        the identified bridge topic.
        
        Args:
            bridge: Bridge topic from find_bridge_topics()
            publisher_analysis: Publisher analysis
            target_analysis: Target analysis
        
        Returns:
            Content brief dict for BACOWR generation_constraints
        """
        logger.info(f"Generating content brief for bridge: {bridge['content_angle']}")
        
        # Merge and prioritize keywords
        merged_keywords = self._merge_keywords(
            publisher_analysis.get('keywords', [])[:10],
            target_analysis.get('keywords', [])[:10]
        )
        
        # Analyze tone alignment
        tone_match = self._analyze_tone_match(publisher_analysis, target_analysis)
        
        brief = {
            'primary_topic': bridge['content_angle'],
            'bridge_strength': bridge['strength'],
            'must_mention_entities': target_analysis.get('primary_entities', [])[:3],
            'context_entities': publisher_analysis.get('entities', {}).get('ORG', [])[:5],
            'semantic_keywords': merged_keywords,
            'content_gaps_to_fill': target_analysis.get('content_gaps', [])[:3],
            'tone_alignment': tone_match,
            'recommended_length': self._estimate_content_length(bridge),
            'key_points': self._extract_key_points(bridge, target_analysis),
            'avoid_topics': self._identify_topics_to_avoid(publisher_analysis, target_analysis)
        }
        
        return brief
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _extract_entities(self, keywords: List[Keyword]) -> Dict[str, List[str]]:
        """Group keywords by entity type."""
        entities = defaultdict(list)
        for kw in keywords:
            if kw.type != "CONCEPT":
                entities[kw.type].append(kw.text)
        return dict(entities)
    
    def _extract_primary_entities(self, keywords: List[Keyword]) -> List[Dict[str, Any]]:
        """Extract most important entities (top 5 by score)."""
        entity_keywords = [kw for kw in keywords if kw.type != "CONCEPT"]
        entity_keywords.sort(key=lambda k: k.score, reverse=True)
        
        return [
            {"text": kw.text, "type": kw.type, "score": kw.score}
            for kw in entity_keywords[:5]
        ]
    
    def _cluster_topics(self, keywords: List[Keyword], n_clusters: int = 5) -> List[Dict[str, Any]]:
        """Cluster keywords into topics using simple frequency-based grouping."""
        # For Phase 2, use simple grouping by score ranges
        # In Phase 3, enhance with actual clustering
        
        sorted_kw = sorted(keywords, key=lambda k: k.score, reverse=True)
        
        topics = []
        cluster_size = max(3, len(sorted_kw) // n_clusters)
        
        for i in range(0, len(sorted_kw), cluster_size):
            cluster = sorted_kw[i:i+cluster_size]
            if cluster:
                topics.append({
                    'main_keyword': cluster[0].text,
                    'keywords': [kw.text for kw in cluster],
                    'avg_score': np.mean([kw.score for kw in cluster]),
                    'size': len(cluster)
                })
        
        return topics[:n_clusters]
    
    def _identify_target_topics(
        self,
        keywords: List[Keyword],
        serp_context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify main topics for target, considering SERP context."""
        topics = self._cluster_topics(keywords, n_clusters=3)
        
        # If SERP context available, align topics with SERP
        if serp_context and 'query' in serp_context:
            query_terms = set(serp_context['query'].lower().split())
            for topic in topics:
                # Calculate overlap with query
                topic_terms = set(' '.join(topic['keywords']).lower().split())
                overlap = len(query_terms & topic_terms)
                topic['query_relevance'] = overlap / max(len(query_terms), 1)
        
        return topics
    
    def _classify_content_type(
        self,
        keywords: List[Keyword],
        text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Classify content type (article, blog, guide, etc.)."""
        # Simple heuristic-based classification
        text_lower = text.lower()
        
        if 'how to' in text_lower or 'guide' in text_lower:
            return 'guide'
        elif 'review' in text_lower or any(kw.text.lower() == 'review' for kw in keywords):
            return 'review'
        elif any(kw.type == 'PERSON' for kw in keywords[:5]):
            return 'interview'
        elif metadata and metadata.get('type') == 'news':
            return 'news'
        else:
            return 'article'
    
    def _extract_authority_signals(self, keywords: List[Keyword], text: str) -> Dict[str, Any]:
        """Extract signals indicating content authority."""
        signals = {
            'expert_entities': [],
            'citations': 0,
            'statistics': 0,
            'authority_keywords': []
        }
        
        # Find expert entities (people, organizations)
        for kw in keywords:
            if kw.type in ['PERSON', 'ORG'] and kw.score > 0.7:
                signals['expert_entities'].append(kw.text)
        
        # Count citations (simple heuristic)
        signals['citations'] = text.count('[') + text.count('(source')
        
        # Count statistics (numbers with %)
        signals['statistics'] = text.count('%')
        
        # Authority keywords
        authority_terms = ['research', 'study', 'according to', 'expert', 'professor']
        for term in authority_terms:
            if term in text.lower():
                signals['authority_keywords'].append(term)
        
        return signals
    
    def _find_link_placement_spots(self, text: str, keywords: List[Keyword]) -> List[Dict[str, Any]]:
        """Identify good locations for link placement."""
        spots = []
        
        # Find paragraphs with high keyword density
        paragraphs = text.split('\n\n')
        
        for i, para in enumerate(paragraphs):
            if len(para) < 50:  # Skip short paragraphs
                continue
            
            # Count keyword mentions
            keyword_count = sum(1 for kw in keywords if kw.text.lower() in para.lower())
            
            if keyword_count >= 2:  # At least 2 keywords
                spots.append({
                    'paragraph_index': i,
                    'keyword_density': keyword_count / max(len(para.split()), 1),
                    'preview': para[:100] + '...',
                    'recommended': keyword_count >= 3
                })
        
        return sorted(spots, key=lambda x: x['keyword_density'], reverse=True)[:5]
    
    def _extract_semantic_themes(self, keywords: List[Keyword]) -> List[str]:
        """Extract high-level semantic themes."""
        # Group related keywords into themes
        themes = set()
        
        for kw in keywords[:20]:  # Top 20 keywords
            # Simple theme extraction based on keyword type and text
            if kw.type == 'PRODUCT':
                themes.add('product_focus')
            elif kw.type in ['PERSON', 'ORG']:
                themes.add('entity_driven')
            elif 'tech' in kw.text.lower() or 'software' in kw.text.lower():
                themes.add('technology')
            elif any(term in kw.text.lower() for term in ['business', 'market', 'revenue']):
                themes.add('business')
        
        return list(themes)
    
    def _build_concept_graph(self, keywords: List[Keyword]) -> Dict[str, List[str]]:
        """Build graph of related concepts."""
        # Simple co-occurrence based graph
        graph = defaultdict(list)
        
        concept_keywords = [kw for kw in keywords if kw.type == 'CONCEPT']
        
        # Connect keywords that appear close together (simple heuristic)
        for i, kw1 in enumerate(concept_keywords[:10]):
            for kw2 in concept_keywords[i+1:min(i+4, len(concept_keywords))]:
                graph[kw1.text].append(kw2.text)
        
        return dict(graph)
    
    def _find_content_gaps(
        self,
        keywords: List[Keyword],
        serp_context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify content gaps vs. SERP competitors."""
        gaps = []
        
        if not serp_context or 'top_10' not in serp_context:
            return gaps
        
        # Extract keywords from this content
        our_keywords = set(kw.text.lower() for kw in keywords)
        
        # Compare with SERP competitors (mock for Phase 2)
        competitor_keywords = set()
        for competitor in serp_context.get('top_10', [])[:3]:
            if 'keywords' in competitor:
                competitor_keywords.update(kw.lower() for kw in competitor['keywords'])
        
        # Find gaps (keywords in competitors but not in ours)
        missing = competitor_keywords - our_keywords
        
        for keyword in list(missing)[:5]:
            gaps.append({
                'keyword': keyword,
                'importance': 'high',  # In Phase 3, calculate actual importance
                'reason': 'Present in top SERP results but missing from content'
            })
        
        return gaps
    
    def _calculate_serp_alignment(
        self,
        keywords: List[Keyword],
        serp_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate how well content aligns with SERP."""
        if not serp_context:
            return {'score': 0.0, 'details': 'No SERP context provided'}
        
        alignment = {
            'score': 0.5,  # Default
            'keyword_coverage': 0.0,
            'topic_match': 'medium',
            'recommendations': []
        }
        
        if 'query' in serp_context:
            query_terms = set(serp_context['query'].lower().split())
            our_terms = set(' '.join(kw.text for kw in keywords).lower().split())
            
            overlap = len(query_terms & our_terms)
            alignment['keyword_coverage'] = overlap / max(len(query_terms), 1)
            alignment['score'] = alignment['keyword_coverage']
        
        return alignment
    
    def _generate_anchor_candidates(self, keywords: List[Keyword]) -> List[Dict[str, Any]]:
        """Generate anchor text candidates."""
        candidates = []
        
        for kw in keywords[:15]:  # Top 15 keywords
            if len(kw.text) <= self.max_anchor_length:
                candidates.append({
                    'text': kw.text,
                    'type': 'exact_match' if kw.type != 'CONCEPT' else 'partial_match',
                    'score': kw.score,
                    'length': len(kw.text)
                })
        
        return sorted(candidates, key=lambda x: x['score'], reverse=True)
    
    def _score_anchor_risk(self, anchor_candidates: List[Dict[str, Any]]) -> Dict[str, float]:
        """Score risk for each anchor text candidate."""
        risk_scores = {}
        
        for anchor in anchor_candidates:
            risk = 0.0
            
            # Exact match entities = higher risk
            if anchor['type'] == 'exact_match':
                risk += 0.4
            
            # Very short anchors = higher risk
            if anchor['length'] < 10:
                risk += 0.2
            
            # Commercial terms = medium risk
            commercial_terms = ['buy', 'best', 'review', 'cheap', 'discount']
            if any(term in anchor['text'].lower() for term in commercial_terms):
                risk += 0.3
            
            risk_scores[anchor['text']] = min(risk, 1.0)
        
        return risk_scores
    
    def _extract_key_concepts(self, keywords: List[Keyword]) -> List[str]:
        """Extract key concepts for content briefs."""
        concepts = [kw.text for kw in keywords if kw.type == 'CONCEPT']
        return concepts[:10]
    
    def _create_semantic_clusters(self, keywords: List[Keyword]) -> List[Dict[str, Any]]:
        """Create semantic clusters of related keywords."""
        return self._cluster_topics(keywords, n_clusters=4)
    
    def _calculate_topic_similarity(self, topic1: Dict[str, Any], topic2: Dict[str, Any]) -> float:
        """Calculate similarity between two topics."""
        # Simple Jaccard similarity on keywords
        kw1 = set(topic1.get('keywords', []))
        kw2 = set(topic2.get('keywords', []))
        
        if not kw1 or not kw2:
            return 0.0
        
        intersection = len(kw1 & kw2)
        union = len(kw1 | kw2)
        
        return intersection / union if union > 0 else 0.0
    
    def _classify_bridge_type(
        self,
        topic1: Dict[str, Any],
        topic2: Dict[str, Any],
        similarity: float
    ) -> str:
        """Classify type of bridge between topics."""
        if similarity > 0.7:
            return 'strong_overlap'
        elif similarity > 0.5:
            return 'moderate_overlap'
        elif similarity > 0.3:
            return 'weak_overlap'
        else:
            return 'tangential'
    
    def _suggest_content_angle(self, topic1: Dict[str, Any], topic2: Dict[str, Any]) -> str:
        """Suggest content angle for bridge."""
        main1 = topic1.get('main_keyword', '')
        main2 = topic2.get('main_keyword', '')
        
        return f"How {main1} relates to {main2}"
    
    def _calculate_bridge_strength(
        self,
        topic1: Dict[str, Any],
        topic2: Dict[str, Any],
        similarity: float
    ) -> float:
        """Calculate overall bridge strength."""
        # Combine similarity with topic scores
        score1 = topic1.get('avg_score', 0.5)
        score2 = topic2.get('avg_score', 0.5)
        
        strength = (similarity * 0.6) + (score1 * 0.2) + (score2 * 0.2)
        return float(strength)
    
    def _merge_keywords(
        self,
        keywords1: List[Dict[str, Any]],
        keywords2: List[Dict[str, Any]]
    ) -> List[str]:
        """Merge and deduplicate keywords from two sources."""
        merged = set()
        
        for kw in keywords1:
            if isinstance(kw, dict):
                merged.add(kw.get('text', str(kw)))
            else:
                merged.add(str(kw))
        
        for kw in keywords2:
            if isinstance(kw, dict):
                merged.add(kw.get('text', str(kw)))
            else:
                merged.add(str(kw))
        
        return list(merged)[:15]
    
    def _analyze_tone_match(
        self,
        publisher_analysis: Dict[str, Any],
        target_analysis: Dict[str, Any]
    ) -> str:
        """Analyze tone compatibility between publisher and target."""
        # Simple heuristic based on content types
        pub_type = publisher_analysis.get('content_type', 'article')
        tgt_type = target_analysis.get('content_type', 'article')
        
        if pub_type == tgt_type:
            return 'strong_match'
        elif pub_type in ['article', 'blog'] and tgt_type in ['article', 'blog']:
            return 'good_match'
        else:
            return 'moderate_match'
    
    def _estimate_content_length(self, bridge: Dict[str, Any]) -> int:
        """Estimate recommended content length."""
        strength = bridge.get('strength', 0.5)
        
        if strength > 0.7:
            return 1500  # Strong bridge = longer content
        elif strength > 0.5:
            return 1000
        else:
            return 750
    
    def _extract_key_points(
        self,
        bridge: Dict[str, Any],
        target_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract key points to cover in content."""
        points = []
        
        # Add bridge angle as first point
        points.append(bridge.get('content_angle', 'Main topic connection'))
        
        # Add top target concepts
        for concept in target_analysis.get('key_concepts', [])[:3]:
            points.append(f"Cover {concept}")
        
        # Add content gaps
        for gap in target_analysis.get('content_gaps', [])[:2]:
            points.append(f"Address gap: {gap.get('keyword', 'missing topic')}")
        
        return points
    
    def _identify_topics_to_avoid(
        self,
        publisher_analysis: Dict[str, Any],
        target_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify topics that should be avoided."""
        avoid = []
        
        # Avoid topics with no overlap
        pub_themes = set(publisher_analysis.get('semantic_themes', []))
        tgt_themes = set(target_analysis.get('semantic_themes', []))  # This might not exist yet
        
        # In Phase 2, keep it simple
        avoid.append('Off-topic competitors')
        avoid.append('Unrelated products')
        
        return avoid
