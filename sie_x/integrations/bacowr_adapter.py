"""
BACOWR Adapter for SIE-X

This adapter integrates SIE-X semantic intelligence into the BACOWR
(Backlink Content Writer) pipeline.

BACOWR Pipeline Integration Points:
1. PageProfiler - Enhanced with semantic analysis
2. IntentAnalyzer - Augmented with topic intelligence
3. serp_research_ext - SERP alignment analysis
4. generation_constraints - Smart content briefs
5. Writer - SEO-optimized content generation
6. QC/AutoFix - Semantic quality checks

Example Usage:
    >>> from sie_x.sdk.python.client import SIEXClient
    >>> from sie_x.integrations.bacowr_adapter import BACOWRAdapter
    >>> 
    >>> client = SIEXClient()
    >>> adapter = BACOWRAdapter(client)
    >>> 
    >>> # In BACOWR pipeline
    >>> enhanced_profile = await adapter.enhance_page_profiler(page_profile)
    >>> bridge = await adapter.find_best_bridge(publisher_url, target_url)
    >>> constraints = await adapter.generate_smart_constraints(backlink_job, bridge)
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from sie_x.sdk.python.client import SIEXClient
from sie_x.transformers.seo_transformer import SEOTransformer

logger = logging.getLogger(__name__)


class BACOWRAdapter:
    """
    Adapter for integrating SIE-X into BACOWR backlink pipeline.
    
    This adapter provides methods to enhance each step of the BACOWR
    workflow with semantic intelligence from SIE-X.
    
    Attributes:
        sie_x: SIE-X client for API communication
        transformer: SEO transformer for analysis
        cache: Optional cache for repeated analyses
    """
    
    def __init__(
        self,
        sie_x_client: SIEXClient,
        use_seo_mode: bool = True,
        cache_enabled: bool = True
    ):
        """
        Initialize BACOWR adapter.
        
        Args:
            sie_x_client: Initialized SIE-X client
            use_seo_mode: Enable SEO transformer (recommended)
            cache_enabled: Enable caching of analyses
        """
        self.sie_x = sie_x_client
        self.transformer = SEOTransformer() if use_seo_mode else None
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Any] = {}
        
        logger.info(f"BACOWRAdapter initialized (SEO mode: {use_seo_mode})")
    
    async def enhance_page_profiler(
        self,
        page_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance PageProfiler with SIE-X semantic analysis.
        
        BACOWR Integration Point: PageProfiler
        Adds semantic entities, topic clusters, and content classification.
        
        Args:
            page_profile: BACOWR PageProfile dict with:
                - url: str
                - text_content: str
                - is_publisher: bool
                - metadata: dict
        
        Returns:
            Enhanced page profile with SIE-X analysis
        """
        url = page_profile.get('url')
        text = page_profile.get('text_content', '')
        is_publisher = page_profile.get('is_publisher', False)
        
        logger.info(f"Enhancing page profile: {url} (publisher: {is_publisher})")
        
        # Check cache
        cache_key = f"profile_{url}"
        if self.cache_enabled and cache_key in self._cache:
            logger.debug(f"Cache hit for {url}")
            return self._cache[cache_key]
        
        # Extract keywords via SIE-X API
        keywords_data = await self.sie_x.extract(
            text=text,
            top_k=20,
            min_confidence=0.3,
            url=url
        )
        
        # Convert to Keyword objects for transformer
        from sie_x.core.models import Keyword
        keywords = [Keyword(**kw) for kw in keywords_data]
        
        # Run SEO transformer analysis
        if self.transformer:
            if is_publisher:
                analysis = await self.transformer.analyze_publisher(
                    text=text,
                    keywords=keywords,
                    url=url,
                    metadata=page_profile.get('metadata')
                )
            else:
                analysis = await self.transformer.analyze_target(
                    text=text,
                    keywords=keywords,
                    url=url,
                    serp_context=page_profile.get('serp_context')
                )
        else:
            # Fallback without transformer
            analysis = {
                'keywords': [kw.model_dump() for kw in keywords],
                'entities': {},
                'topics': []
            }
        
        # Enrich page profile
        enhanced = page_profile.copy()
        enhanced['sie_x_analysis'] = analysis
        enhanced['semantic_entities'] = analysis.get('entities', {})
        enhanced['topic_clusters'] = analysis.get('topics', [])
        enhanced['content_type'] = analysis.get('content_type', 'unknown')
        enhanced['analyzed_at'] = datetime.now().isoformat()
        
        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = enhanced
        
        logger.info(f"Profile enhanced: {len(analysis.get('keywords', []))} keywords, "
                   f"{len(analysis.get('topics', []))} topics")
        
        return enhanced
    
    async def enhance_intent_analysis(
        self,
        intent_data: Dict[str, Any],
        sie_x_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance intent analysis with SIE-X semantic insights.
        
        BACOWR Integration Point: IntentAnalyzer
        Adds semantic alignment scoring and topic relevance.
        
        Args:
            intent_data: BACOWR intent analysis with:
                - intent_type: str (informational, transactional, etc.)
                - keywords: List[str]
                - user_goal: str
            sie_x_analysis: SIE-X analysis from enhance_page_profiler()
        
        Returns:
            Enhanced intent data with semantic insights
        """
        logger.info("Enhancing intent analysis with semantic insights")
        
        enhanced = intent_data.copy()
        
        # Calculate semantic alignment
        intent_keywords = set(kw.lower() for kw in intent_data.get('keywords', []))
        sie_x_keywords = set(
            kw['text'].lower() 
            for kw in sie_x_analysis.get('keywords', [])
        )
        
        overlap = len(intent_keywords & sie_x_keywords)
        total = len(intent_keywords | sie_x_keywords)
        alignment = overlap / total if total > 0 else 0.0
        
        enhanced['semantic_alignment'] = {
            'score': alignment,
            'overlap_keywords': list(intent_keywords & sie_x_keywords),
            'missing_keywords': list(intent_keywords - sie_x_keywords)
        }
        
        # Add topic relevance
        enhanced['topic_relevance'] = sie_x_analysis.get('serp_alignment', {})
        
        # Add content gaps
        enhanced['content_opportunities'] = sie_x_analysis.get('content_gaps', [])
        
        logger.info(f"Intent enhanced: semantic alignment = {alignment:.2f}")
        
        return enhanced
    
    async def find_best_bridge(
        self,
        publisher_url: str,
        target_url: str,
        publisher_text: Optional[str] = None,
        target_text: Optional[str] = None,
        serp_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best semantic bridge between publisher and target.
        
        BACOWR Integration Point: Custom bridge finder
        Identifies optimal content angle for natural link placement.
        
        Args:
            publisher_url: Publisher page URL
            target_url: Target page URL  
            publisher_text: Publisher content (optional, will fetch if None)
            target_text: Target content (optional, will fetch if None)
            serp_context: SERP data for target query
        
        Returns:
            Best bridge topic dict or None if no good bridge found
        """
        logger.info(f"Finding bridge: {publisher_url} -> {target_url}")
        
        if not self.transformer:
            logger.warning("SEO transformer not enabled, cannot find bridges")
            return None
        
        # Analyze publisher
        if publisher_text:
            pub_keywords = await self.sie_x.extract(publisher_text, top_k=20, url=publisher_url)
            from sie_x.core.models import Keyword
            pub_kw_objects = [Keyword(**kw) for kw in pub_keywords]
            
            pub_analysis = await self.transformer.analyze_publisher(
                text=publisher_text,
                keywords=pub_kw_objects,
                url=publisher_url
            )
        else:
            # Fetch from URL
            pub_analysis_result = await self.sie_x.analyze_url(publisher_url, top_k=20)
            # Simplified analysis without full transformer
            pub_analysis = {
                'keywords': pub_analysis_result,
                'topics': [],
                'entities': {}
            }
        
        # Analyze target
        if target_text:
            tgt_keywords = await self.sie_x.extract(target_text, top_k=20, url=target_url)
            from sie_x.core.models import Keyword
            tgt_kw_objects = [Keyword(**kw) for kw in tgt_keywords]
            
            tgt_analysis = await self.transformer.analyze_target(
                text=target_text,
                keywords=tgt_kw_objects,
                url=target_url,
                serp_context=serp_context
            )
        else:
            # Fetch from URL
            tgt_analysis_result = await self.sie_x.analyze_url(target_url, top_k=20)
            tgt_analysis = {
                'keywords': tgt_analysis_result,
                'target_topics': [],
                'primary_entities': []
            }
        
        # Find bridges
        bridges = self.transformer.find_bridge_topics(pub_analysis, tgt_analysis)
        
        if not bridges:
            logger.warning("No bridge topics found")
            return None
        
        best_bridge = bridges[0]  # Highest strength
        logger.info(f"Best bridge found: {best_bridge['content_angle']} (strength: {best_bridge['strength']:.2f})")
        
        # Add full analyses to bridge
        best_bridge['publisher_analysis'] = pub_analysis
        best_bridge['target_analysis'] = tgt_analysis
        
        return best_bridge
    
    async def generate_smart_constraints(
        self,
        backlink_job: Dict[str, Any],
        bridge: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate smart generation_constraints for BACOWR Writer.

        BACOWR Integration Point: generation_constraints
        Creates detailed content requirements based on semantic analysis.

        Args:
            backlink_job: BACOWR BacklinkJob dict
            bridge: Bridge topic from find_best_bridge()

        Returns:
            Enhanced generation_constraints dict for Writer
        """
        logger.info("Generating smart constraints from bridge")

        if not self.transformer:
            logger.warning("SEO transformer not enabled, using basic constraints")
            return backlink_job.get('generation_constraints', {})

        # Generate content brief
        content_brief = self.transformer.generate_content_brief(
            bridge=bridge,
            publisher_analysis=bridge.get('publisher_analysis', {}),
            target_analysis=bridge.get('target_analysis', {})
        )

        # Build constraints
        constraints = {
            'content_requirements': {
                'primary_topic': content_brief['primary_topic'],
                'semantic_keywords': content_brief['semantic_keywords'],
                'must_mention_entities': content_brief['must_mention_entities'],
                'context_entities': content_brief['context_entities'],
                'key_points': content_brief.get('key_points', []),
                'tone': content_brief['tone_alignment'],
                'length': {
                    'min': content_brief['recommended_length'] - 200,
                    'max': content_brief['recommended_length'] + 300,
                    'target': content_brief['recommended_length']
                }
            },
            'link_requirements': {
                'anchor_text_options': bridge.get('target_analysis', {}).get('anchor_candidates', [])[:5],
                'placement_context': self._extract_placement_context(bridge),
                'risk_scores': bridge.get('target_analysis', {}).get('anchor_risk_scores', {}),
                'preferred_anchor_type': 'partial_match'  # Safer than exact match
            },
            'quality_criteria': {
                'topic_relevance_min': 0.7,
                'semantic_coherence_min': 0.8,
                'entity_coverage_min': 0.6,
                'keyword_density_max': 0.03,  # 3% max
                'readability_min': 60  # Flesch reading ease
            },
            'avoid': {
                'topics': content_brief.get('avoid_topics', []),
                'over_optimization': True,
                'keyword_stuffing': True
            },
            'sie_x_metadata': {
                'bridge_strength': bridge['strength'],
                'bridge_type': bridge['bridge_type'],
                'generated_at': datetime.now().isoformat()
            }
        }

        logger.info(f"Smart constraints generated: {len(content_brief['semantic_keywords'])} keywords, "
                   f"{len(content_brief['must_mention_entities'])} entities")

        return constraints

    async def generate_bacowr_extensions(
        self,
        bridge: Dict[str, Any],
        serp_context: Optional[Dict[str, Any]] = None,
        trust_level: str = "T1"
    ) -> Dict[str, Any]:
        """
        Generate BACOWR v2 extension fields for BacklinkArticleOutput.

        This method produces the exact JSON schema fields required by BACOWR
        preflight logic, including bridge_type, intent_extension, links_extension,
        qc_extension, and serp_research_extension.

        Args:
            bridge: Bridge topic from find_best_bridge()
            serp_context: SERP analysis data (optional)
            trust_level: Trust policy level (T1=public, T2=academic, T3=industry, T4=media)

        Returns:
            Dict with BACOWR v2 extension fields:
            - intent_extension: Intent alignment and bridge recommendation
            - links_extension: Link placement, trust policy, compliance
            - qc_extension: Anchor risk, readability, QC thresholds
            - serp_research_extension: SERP analysis (if serp_context provided)
        """
        logger.info(f"Generating BACOWR v2 extensions (trust_level={trust_level})")

        pub_analysis = bridge.get('publisher_analysis', {})
        tgt_analysis = bridge.get('target_analysis', {})

        # Determine bridge type based on semantic strength
        bridge_strength = bridge.get('strength', 0.0)
        if bridge_strength >= 0.8:
            bridge_type = "strong"  # Direct semantic overlap
        elif bridge_strength >= 0.5:
            bridge_type = "pivot"   # Intermediate topic connection
        else:
            bridge_type = "wrapper" # Weak connection, needs context

        # Generate intent_extension
        intent_extension = self._generate_intent_extension(
            bridge_type=bridge_type,
            bridge=bridge,
            serp_context=serp_context,
            tgt_analysis=tgt_analysis
        )

        # Generate links_extension
        links_extension = self._generate_links_extension(
            bridge_type=bridge_type,
            bridge=bridge,
            trust_level=trust_level,
            pub_analysis=pub_analysis,
            tgt_analysis=tgt_analysis
        )

        # Generate qc_extension
        qc_extension = self._generate_qc_extension(
            bridge=bridge,
            tgt_analysis=tgt_analysis
        )

        # Generate serp_research_extension (if SERP data available)
        serp_research_extension = None
        if serp_context:
            serp_research_extension = self._generate_serp_research_extension(
                serp_context=serp_context,
                tgt_analysis=tgt_analysis
            )

        extensions = {
            'intent_extension': intent_extension,
            'links_extension': links_extension,
            'qc_extension': qc_extension
        }

        if serp_research_extension:
            extensions['serp_research_extension'] = serp_research_extension

        logger.info(f"BACOWR extensions generated: bridge_type={bridge_type}, "
                   f"intent_alignment={intent_extension['intent_alignment']:.2f}")

        return extensions

    def _generate_intent_extension(
        self,
        bridge_type: str,
        bridge: Dict[str, Any],
        serp_context: Optional[Dict[str, Any]],
        tgt_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate intent_extension for BACOWR."""
        # Determine SERP intent (informational, transactional, navigational, commercial)
        serp_intent_primary = "informational"  # Default
        serp_intent_secondary = None

        if serp_context:
            # Analyze SERP features to determine intent
            features = serp_context.get('features', [])
            if 'shopping_results' in features or 'ads' in features:
                serp_intent_primary = "commercial"
            elif 'knowledge_panel' in features or 'featured_snippet' in features:
                serp_intent_primary = "informational"

            # Secondary intent
            if len(features) > 1:
                serp_intent_secondary = "navigational" if 'sitelinks' in features else None

        # Calculate intent alignment (SERP vs publisher vs target vs anchor)
        # Higher alignment = better variabelgifte (variable marriage)
        tgt_topics = tgt_analysis.get('target_topics', [])
        bridge_topics = bridge.get('shared_topics', [])

        topic_overlap = len(set(bridge_topics) & set(tgt_topics)) / max(len(tgt_topics), 1)
        intent_alignment = min(bridge.get('strength', 0.5) + (topic_overlap * 0.3), 1.0)

        return {
            'serp_intent_primary': serp_intent_primary,
            'serp_intent_secondary': serp_intent_secondary,
            'intent_alignment': round(intent_alignment, 3),
            'recommended_bridge_type': bridge_type,
            'alignment_notes': self._get_alignment_notes(intent_alignment, bridge_type)
        }

    def _generate_links_extension(
        self,
        bridge_type: str,
        bridge: Dict[str, Any],
        trust_level: str,
        pub_analysis: Dict[str, Any],
        tgt_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate links_extension for BACOWR."""
        # Get anchor candidates and calculate risk
        anchor_candidates = tgt_analysis.get('anchor_candidates', [])
        anchor_risk_scores = tgt_analysis.get('anchor_risk_scores', {})

        # Select best anchor based on risk and bridge type
        if bridge_type == "strong":
            # Can use more exact anchors
            preferred_anchor = anchor_candidates[0] if anchor_candidates else "learn more"
            anchor_swap = False
        elif bridge_type == "pivot":
            # Use partial match
            preferred_anchor = anchor_candidates[1] if len(anchor_candidates) > 1 else anchor_candidates[0] if anchor_candidates else "more information"
            anchor_swap = False
        else:  # wrapper
            # Use generic anchor, allow swap
            preferred_anchor = "read more"
            anchor_swap = True

        # Determine placement based on publisher content structure
        link_placement_spots = pub_analysis.get('link_placement_spots', [])
        if link_placement_spots:
            placement = link_placement_spots[0].get('location', 'middle_paragraph')
        else:
            placement = 'middle_paragraph'  # Safe default

        # Trust policy based on level
        trust_policies = {
            'T1': {'do_follow': True, 'sponsored': False, 'ugc': False},   # Public/blogs
            'T2': {'do_follow': True, 'sponsored': False, 'ugc': False},   # Academic
            'T3': {'do_follow': True, 'sponsored': False, 'ugc': False},   # Industry
            'T4': {'do_follow': True, 'sponsored': True, 'ugc': False}     # Media (may sponsor)
        }
        trust_policy = trust_policies.get(trust_level, trust_policies['T1'])

        # Compliance checks
        compliance = {
            'lsi_quality_pass': self._check_lsi_quality(bridge, tgt_analysis),
            'near_window_pass': True,  # Would check actual link density
            'publisher_fit_pass': self._check_publisher_fit(pub_analysis, bridge),
            'anchor_risk_acceptable': self._check_anchor_risk(preferred_anchor, anchor_risk_scores)
        }

        return {
            'bridge_type': bridge_type,
            'anchor_text': preferred_anchor,
            'anchor_swap': anchor_swap,
            'placement': placement,
            'trust_policy': trust_policy,
            'compliance': compliance
        }

    def _generate_qc_extension(
        self,
        bridge: Dict[str, Any],
        tgt_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate qc_extension for BACOWR."""
        # Get anchor risk from target analysis
        anchor_candidates = tgt_analysis.get('anchor_candidates', [])
        anchor_risk_scores = tgt_analysis.get('anchor_risk_scores', {})

        # Calculate overall anchor risk (0-1, higher = riskier)
        if anchor_candidates and anchor_risk_scores:
            avg_risk = sum(anchor_risk_scores.values()) / len(anchor_risk_scores)
            anchor_risk = "high" if avg_risk > 0.7 else "medium" if avg_risk > 0.4 else "low"
        else:
            anchor_risk = "low"  # No anchors = generic link = low risk

        # Estimate readability (Flesch Reading Ease approximation)
        # Would calculate from actual content, using 65 as baseline
        readability = 65.0

        return {
            'anchor_risk': anchor_risk,
            'readability': readability,
            'thresholds_version': 'v2.1',
            'qc_checks': {
                'semantic_coherence': bridge.get('strength', 0.5),
                'topic_relevance': min(bridge.get('strength', 0.5) + 0.2, 1.0),
                'entity_coverage': 0.7,  # Would calculate from actual content
                'keyword_density': 0.02  # Would calculate from actual content
            }
        }

    def _generate_serp_research_extension(
        self,
        serp_context: Dict[str, Any],
        tgt_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate serp_research_extension for BACOWR."""
        # Extract SERP features and competition
        serp_features = serp_context.get('features', [])
        top_results = serp_context.get('organic_results', [])[:10]

        # Analyze competition
        competition_level = "high" if len(top_results) >= 10 else "medium" if len(top_results) >= 5 else "low"

        # Calculate SERP alignment
        tgt_topics = set(tgt_analysis.get('target_topics', []))
        serp_topics = set()
        for result in top_results:
            # Would extract topics from SERP snippets
            snippet = result.get('snippet', '')
            # Simplified: just use tgt_topics for demo
            serp_topics.update(tgt_topics)

        serp_alignment = len(tgt_topics & serp_topics) / max(len(tgt_topics), 1)

        return {
            'query': serp_context.get('query', ''),
            'serp_features': serp_features,
            'competition_level': competition_level,
            'serp_alignment': round(serp_alignment, 3),
            'top_ranking_topics': list(serp_topics)[:10],
            'content_gaps': tgt_analysis.get('content_gaps', []),
            'ranking_difficulty': self._estimate_ranking_difficulty(competition_level, serp_features)
        }

    def _get_alignment_notes(self, alignment: float, bridge_type: str) -> str:
        """Generate alignment notes for intent_extension."""
        if alignment >= 0.8:
            return f"Excellent alignment with {bridge_type} bridge - strong variabelgifte"
        elif alignment >= 0.6:
            return f"Good alignment with {bridge_type} bridge - acceptable variabelgifte"
        elif alignment >= 0.4:
            return f"Moderate alignment with {bridge_type} bridge - may need content adjustments"
        else:
            return f"Weak alignment with {bridge_type} bridge - consider different approach"

    def _check_lsi_quality(self, bridge: Dict[str, Any], tgt_analysis: Dict[str, Any]) -> bool:
        """Check LSI quality - semantic keywords present."""
        shared_topics = bridge.get('shared_topics', [])
        return len(shared_topics) >= 2  # At least 2 shared semantic topics

    def _check_publisher_fit(self, pub_analysis: Dict[str, Any], bridge: Dict[str, Any]) -> bool:
        """Check if publisher is a good fit for the bridge topic."""
        pub_topics = pub_analysis.get('topics', [])
        bridge_topics = bridge.get('shared_topics', [])
        overlap = len(set(pub_topics) & set(bridge_topics))
        return overlap >= 1  # At least one overlapping topic

    def _check_anchor_risk(self, anchor: str, risk_scores: Dict[str, float]) -> bool:
        """Check if anchor risk is acceptable."""
        risk = risk_scores.get(anchor, 0.5)  # Default medium risk
        return risk < 0.7  # Acceptable if < 0.7

    def _estimate_ranking_difficulty(self, competition: str, features: List[str]) -> str:
        """Estimate ranking difficulty from SERP analysis."""
        if competition == "high" and len(features) > 5:
            return "very_hard"
        elif competition == "high":
            return "hard"
        elif competition == "medium":
            return "moderate"
        else:
            return "easy"
    
    async def validate_generated_content(
        self,
        content: str,
        constraints: Dict[str, Any],
        target_url: str
    ) -> Dict[str, Any]:
        """
        Validate generated content using SIE-X semantic analysis.
        
        BACOWR Integration Point: QC/AutoFix
        Checks if generated content meets semantic requirements.
        
        Args:
            content: Generated content text
            constraints: Smart constraints from generate_smart_constraints()
            target_url: Target URL for link
        
        Returns:
            Validation result with issues and suggestions
        """
        logger.info("Validating generated content with SIE-X")
        
        # Extract keywords from generated content
        content_keywords = await self.sie_x.extract(content, top_k=30)
        
        validation = {
            'passed': True,
            'score': 1.0,
            'issues': [],
            'suggestions': [],
            'metrics': {}
        }
        
        # Check keyword coverage
        required_keywords = set(
            kw.lower() 
            for kw in constraints.get('content_requirements', {}).get('semantic_keywords', [])
        )
        found_keywords = set(kw['text'].lower() for kw in content_keywords)
        
        keyword_coverage = len(required_keywords & found_keywords) / len(required_keywords) if required_keywords else 1.0
        validation['metrics']['keyword_coverage'] = keyword_coverage
        
        if keyword_coverage < 0.5:
            validation['passed'] = False
            validation['issues'].append({
                'type': 'low_keyword_coverage',
                'severity': 'high',
                'message': f'Only {keyword_coverage:.0%} of required keywords present',
                'missing_keywords': list(required_keywords - found_keywords)[:5]
            })
            validation['suggestions'].append('Add more semantic keywords naturally throughout the content')
        
        # Check entity coverage
        required_entities = constraints.get('content_requirements', {}).get('must_mention_entities', [])
        entity_coverage = sum(
            1 for ent in required_entities
            if ent['text'].lower() in content.lower()
        ) / len(required_entities) if required_entities else 1.0
        
        validation['metrics']['entity_coverage'] = entity_coverage
        
        if entity_coverage < 0.6:
            validation['passed'] = False
            validation['issues'].append({
                'type': 'missing_entities',
                'severity': 'medium',
                'message': f'Only {entity_coverage:.0%} of required entities mentioned'
            })
        
        # Check length
        content_length = len(content.split())
        target_length = constraints.get('content_requirements', {}).get('length', {}).get('target', 1000)
        length_diff = abs(content_length - target_length) / target_length
        
        validation['metrics']['length'] = content_length
        validation['metrics']['length_score'] = max(0, 1 - length_diff)
        
        if length_diff > 0.3:  # More than 30% off
            validation['issues'].append({
                'type': 'length_mismatch',
                'severity': 'low',
                'message': f'Content is {content_length} words, target is {target_length}'
            })
        
        # Calculate overall score
        validation['score'] = (
            keyword_coverage * 0.4 +
            entity_coverage * 0.3 +
            validation['metrics']['length_score'] * 0.3
        )
        
        logger.info(f"Validation complete: score={validation['score']:.2f}, passed={validation['passed']}")
        
        return validation
    
    def _extract_placement_context(self, bridge: Dict[str, Any]) -> str:
        """Extract context for link placement."""
        return (
            f"Natural link placement in content about {bridge['content_angle']}. "
            f"Bridge strength: {bridge['strength']:.2f}"
        )
    
    def clear_cache(self):
        """Clear the adapter's cache."""
        self._cache.clear()
        logger.info("Adapter cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'enabled': self.cache_enabled,
            'size': len(self._cache),
            'keys': list(self._cache.keys())
        }
