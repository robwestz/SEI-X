"""
Simple Semantic Engine for SIE-X keyword extraction.

This is the core extraction engine that combines:
- spaCy for NER and linguistic analysis
- Sentence Transformers for semantic embeddings
- NetworkX + PageRank for graph-based ranking
"""

import spacy
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
import logging
import hashlib
import time
from collections import defaultdict

from .models import Keyword

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleSemanticEngine:
    """
    Semantic keyword extraction engine using graph-based ranking.
    
    Features:
    - Named Entity Recognition (NER) via spaCy
    - Semantic similarity via Sentence Transformers
    - Graph-based ranking via PageRank
    - Embedding caching for performance
    
    Example:
    ```python
    engine = SimpleSemanticEngine()
    keywords = engine.extract("Your text here...", top_k=10)
    ```
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        spacy_model: str = "en_core_web_sm"
    ):
        """
        Initialize the semantic engine.
        
        Args:
            model_name: Sentence transformer model to use
            spacy_model: spaCy model to use for NER
        """
        logger.info(f"Initializing SimpleSemanticEngine with {model_name}")
        
        # Load models
        try:
            self.embedder = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            raise
        
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise
        
        # Embedding cache for performance
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.model_name = model_name
        self.spacy_model = spacy_model
        
        logger.info("SimpleSemanticEngine initialized successfully")
    
    def extract(
        self,
        text: str,
        top_k: int = 10,
        min_confidence: float = 0.3,
        include_entities: bool = True,
        include_concepts: bool = True
    ) -> List[Keyword]:
        """
        Extract keywords from text using semantic similarity and graph ranking.
        
        Args:
            text: Input text to extract keywords from
            top_k: Maximum number of keywords to return
            min_confidence: Minimum confidence threshold (0-1)
            include_entities: Include named entities (PERSON, ORG, LOC)
            include_concepts: Include concept keywords (noun phrases)
        
        Returns:
            List of Keyword objects, sorted by score (highest first)
        
        Raises:
            ValueError: If text is empty or invalid
        """
        # Validate input
        text = text.strip()
        if not text:
            logger.warning("Empty text provided, returning empty list")
            return []
        
        logger.info(f"Extracting keywords from text ({len(text)} chars)")
        start_time = time.time()
        
        try:
            # Step 1: Generate keyword candidates
            candidates = self._generate_candidates(
                text,
                include_entities=include_entities,
                include_concepts=include_concepts
            )
            
            if not candidates:
                logger.warning("No candidates generated")
                return []
            
            logger.info(f"Generated {len(candidates)} candidates")
            
            # Step 2: Get embeddings for candidates
            candidate_list = list(candidates.values())
            texts = [kw.text for kw in candidate_list]
            embeddings = self._get_embeddings(texts)
            
            # Step 3: Build similarity graph
            graph = self._build_graph(candidate_list, embeddings)
            
            # Step 4: Rank keywords using PageRank
            ranked_keywords = self._rank_keywords(graph, candidate_list)
            
            # Step 5: Filter by confidence and return top-k
            filtered = [
                kw for kw in ranked_keywords
                if kw.confidence >= min_confidence
            ]
            
            result = filtered[:top_k]
            
            elapsed = time.time() - start_time
            logger.info(
                f"Extracted {len(result)} keywords in {elapsed:.3f}s "
                f"(from {len(candidates)} candidates)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during extraction: {e}", exc_info=True)
            raise
    
    def _generate_candidates(
        self,
        text: str,
        include_entities: bool = True,
        include_concepts: bool = True
    ) -> Dict[str, Keyword]:
        """
        Generate keyword candidates from text using NER and noun phrases.
        
        Args:
            text: Input text
            include_entities: Include named entities
            include_concepts: Include noun phrases
        
        Returns:
            Dict mapping normalized text to Keyword objects
        """
        candidates: Dict[str, Keyword] = {}
        doc = self.nlp(text)
        
        # Extract named entities
        if include_entities:
            for ent in doc.ents:
                normalized = ent.text.lower().strip()
                if len(normalized) < 2:  # Skip single characters
                    continue
                
                if normalized in candidates:
                    # Update existing keyword
                    candidates[normalized].count += 1
                    candidates[normalized].positions.append((ent.start_char, ent.end_char))
                else:
                    # Create new keyword
                    candidates[normalized] = Keyword(
                        text=ent.text,
                        score=0.0,  # Will be set during ranking
                        type=ent.label_,
                        count=1,
                        confidence=1.0,
                        positions=[(ent.start_char, ent.end_char)],
                        metadata={"source": "entity"}
                    )
        
        # Extract noun phrases (concepts)
        if include_concepts:
            for chunk in doc.noun_chunks:
                # Clean up the noun phrase
                chunk_text = chunk.text.strip()
                normalized = chunk_text.lower()
                
                # Filter out very short or very long phrases
                if len(normalized) < 3 or len(normalized) > 50:
                    continue
                
                # Skip if it's just stopwords or punctuation
                if all(token.is_stop or token.is_punct for token in chunk):
                    continue
                
                if normalized in candidates:
                    candidates[normalized].count += 1
                    candidates[normalized].positions.append((chunk.start_char, chunk.end_char))
                else:
                    candidates[normalized] = Keyword(
                        text=chunk_text,
                        score=0.0,
                        type="CONCEPT",
                        count=1,
                        confidence=1.0,
                        positions=[(chunk.start_char, chunk.end_char)],
                        metadata={"source": "noun_phrase"}
                    )
        
        return candidates
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for texts with caching.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            NumPy array of embeddings (shape: [len(texts), embedding_dim])
        """
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                embeddings.append(None)
        
        # Embed uncached texts in batch
        if texts_to_embed:
            logger.debug(f"Embedding {len(texts_to_embed)} new texts")
            new_embeddings = self.embedder.encode(
                texts_to_embed,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Update cache and results
            for idx, text, emb in zip(indices_to_embed, texts_to_embed, new_embeddings):
                cache_key = self._get_cache_key(text)
                self.embedding_cache[cache_key] = emb
                embeddings[idx] = emb
        
        return np.array(embeddings)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _build_graph(
        self,
        keywords: List[Keyword],
        embeddings: np.ndarray,
        similarity_threshold: float = 0.3
    ) -> nx.Graph:
        """
        Build similarity graph from keyword embeddings.
        
        Args:
            keywords: List of Keyword objects
            embeddings: Embedding matrix
            similarity_threshold: Minimum similarity to create edge
        
        Returns:
            NetworkX graph with keywords as nodes and similarities as edges
        """
        graph = nx.Graph()
        
        # Add nodes
        for i, keyword in enumerate(keywords):
            graph.add_node(i, keyword=keyword)
        
        # Add edges based on cosine similarity (vectorized for performance)
        n = len(keywords)
        if n < 2:
            return graph

        # Normalize embeddings to unit vectors for efficient cosine similarity calculation
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Use a small epsilon to avoid division by zero for zero-vectors
        normalized_embeddings = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms!=0)

        # Compute cosine similarity matrix
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

        # Add edges from the upper triangle of the similarity matrix
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i, j]
                if similarity > similarity_threshold:
                    graph.add_edge(i, j, weight=float(similarity))
        
        logger.debug(f"Built graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        return graph
    
    def _rank_keywords(
        self,
        graph: nx.Graph,
        keywords: List[Keyword]
    ) -> List[Keyword]:
        """
        Rank keywords using PageRank combined with frequency.
        
        Args:
            graph: Similarity graph
            keywords: List of Keyword objects
        
        Returns:
            List of keywords sorted by score (highest first)
        """
        if graph.number_of_nodes() == 0:
            return []
        
        # Run PageRank
        try:
            pagerank_scores = nx.pagerank(graph, weight='weight')
        except Exception as e:
            logger.warning(f"PageRank failed: {e}, using uniform scores")
            pagerank_scores = {i: 1.0 / len(keywords) for i in range(len(keywords))}
        
        # Combine PageRank with frequency
        max_count = max(kw.count for kw in keywords)
        
        for i, keyword in enumerate(keywords):
            pr_score = pagerank_scores.get(i, 0.0)
            freq_score = keyword.count / max_count
            
            # Combined score: 70% PageRank, 30% frequency
            combined_score = 0.7 * pr_score + 0.3 * freq_score
            
            keyword.score = float(combined_score)
            keyword.confidence = float(pr_score)  # Use PageRank as confidence
        
        # Sort by score (descending)
        ranked = sorted(keywords, key=lambda k: k.score, reverse=True)
        
        return ranked
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self.model_name,
            "spacy_model": self.spacy_model,
            "cache_size": len(self.embedding_cache),
        }
