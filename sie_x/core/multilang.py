"""
SIE-X Multi-Language Support

Automatic language detection and multi-language processing.
Supports dynamic model loading based on detected language.

Supported languages (with spaCy models):
- English: en_core_web_sm
- Spanish: es_core_news_sm
- French: fr_core_news_sm
- German: de_core_news_sm
- Italian: it_core_news_sm
- Portuguese: pt_core_news_sm
- Dutch: nl_core_news_sm
- Greek: el_core_news_sm
- Norwegian: nb_core_news_sm
- Lithuanian: lt_core_news_sm

Example:
    >>> from sie_x.core.multilang import MultiLangEngine
    >>>
    >>> engine = MultiLangEngine()
    >>> keywords = engine.extract("Hola mundo, esto es un ejemplo")  # Auto-detects Spanish
    >>> print(keywords[0].text)  # "mundo"
"""

from typing import List, Dict, Optional, Any
import logging
from functools import lru_cache

from sie_x.core.models import Keyword, ExtractionOptions
from sie_x.core.simple_engine import SimpleExtractionEngine

logger = logging.getLogger(__name__)


# Language code to spaCy model mapping
SPACY_MODELS = {
    'en': 'en_core_web_sm',
    'es': 'es_core_news_sm',
    'fr': 'fr_core_news_sm',
    'de': 'de_core_news_sm',
    'it': 'it_core_news_sm',
    'pt': 'pt_core_news_sm',
    'nl': 'nl_core_news_sm',
    'el': 'el_core_news_sm',
    'nb': 'nb_core_news_sm',  # Norwegian Bokmål
    'lt': 'lt_core_news_sm'
}

# Default fallback language
DEFAULT_LANGUAGE = 'en'


class LanguageDetector:
    """
    Simple language detector using character n-grams and word patterns.

    For production use, consider integrating:
    - langdetect: pip install langdetect
    - fasttext: pip install fasttext
    - lingua-language-detector: pip install lingua-language-detector
    """

    def __init__(self):
        """Initialize language detector."""
        # Common words for basic detection
        self.language_patterns = {
            'en': ['the', 'is', 'and', 'to', 'of', 'in', 'that', 'it', 'with'],
            'es': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'por', 'los'],
            'fr': ['le', 'de', 'un', 'et', 'être', 'à', 'il', 'avoir', 'ne', 'je'],
            'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'],
            'it': ['il', 'di', 'e', 'la', 'che', 'è', 'un', 'per', 'in', 'non'],
            'pt': ['o', 'de', 'e', 'a', 'que', 'do', 'em', 'um', 'os', 'no'],
            'nl': ['de', 'het', 'een', 'en', 'van', 'in', 'op', 'te', 'dat', 'is'],
            'el': ['το', 'και', 'τη', 'της', 'ο', 'στο', 'που', 'με', 'για', 'από'],
            'nb': ['og', 'i', 'det', 'er', 'til', 'som', 'på', 'en', 'av', 'med'],
            'lt': ['ir', 'yra', 'kad', 'būti', 'su', 'į', 'iš', 'nei', 'ar', 'tai']
        }

        logger.info("LanguageDetector initialized (pattern-based)")

    def detect(self, text: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Detect language of text.

        Args:
            text: Input text
            top_n: Number of top language candidates to return

        Returns:
            List of dicts with 'language' and 'confidence' (sorted by confidence)
        """
        if not text or len(text.strip()) < 10:
            # Too short to detect reliably
            return [{'language': DEFAULT_LANGUAGE, 'confidence': 0.5}]

        # Lowercase and tokenize
        words = text.lower().split()

        # Count matches for each language
        scores = {}
        for lang, patterns in self.language_patterns.items():
            matches = sum(1 for word in words if word in patterns)
            # Normalize by text length
            scores[lang] = matches / len(words) if words else 0

        # Sort by score
        sorted_langs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for lang, score in sorted_langs[:top_n]:
            if score > 0:
                results.append({
                    'language': lang,
                    'confidence': min(score * 10, 1.0)  # Scale to 0-1
                })

        # Fallback to English if no matches
        if not results:
            results.append({'language': DEFAULT_LANGUAGE, 'confidence': 0.3})

        logger.debug(f"Detected languages: {results}")
        return results


class MultiLangEngine:
    """
    Multi-language extraction engine with automatic language detection.

    Automatically detects input language and uses appropriate spaCy model.
    Falls back to English if language not supported or detection fails.

    Attributes:
        detector: Language detector
        engines: Cache of SimpleExtractionEngine instances per language
        default_language: Fallback language (default: 'en')
        auto_detect: Whether to auto-detect language (default: True)
    """

    def __init__(
        self,
        default_language: str = DEFAULT_LANGUAGE,
        auto_detect: bool = True,
        cache_size: int = 5
    ):
        """
        Initialize multi-language engine.

        Args:
            default_language: Default language code (e.g., 'en', 'es')
            auto_detect: Enable automatic language detection
            cache_size: Maximum number of language engines to keep in memory
        """
        self.default_language = default_language
        self.auto_detect = auto_detect
        self.detector = LanguageDetector() if auto_detect else None

        # Cache of engines per language
        self.engines: Dict[str, SimpleExtractionEngine] = {}
        self.cache_size = cache_size

        # Statistics
        self.stats = {
            'total_extractions': 0,
            'languages_detected': {},
            'cache_hits': 0,
            'cache_misses': 0
        }

        logger.info(f"MultiLangEngine initialized: default={default_language}, "
                   f"auto_detect={auto_detect}, cache_size={cache_size}")

    def _get_or_create_engine(self, language: str) -> SimpleExtractionEngine:
        """
        Get cached engine or create new one for language.

        Args:
            language: Language code

        Returns:
            SimpleExtractionEngine for the language
        """
        # Check cache
        if language in self.engines:
            self.stats['cache_hits'] += 1
            logger.debug(f"Engine cache hit for language: {language}")
            return self.engines[language]

        self.stats['cache_misses'] += 1

        # Check if we need to evict oldest engine (simple FIFO)
        if len(self.engines) >= self.cache_size:
            # Remove first (oldest) engine
            oldest = list(self.engines.keys())[0]
            logger.info(f"Evicting engine from cache: {oldest}")
            del self.engines[oldest]

        # Get spaCy model name
        spacy_model = SPACY_MODELS.get(language)

        if not spacy_model:
            logger.warning(f"No spaCy model for language '{language}', using default '{self.default_language}'")
            spacy_model = SPACY_MODELS.get(self.default_language, 'en_core_web_sm')
            language = self.default_language

        # Create new engine
        try:
            logger.info(f"Creating new engine for language: {language} (model: {spacy_model})")
            engine = SimpleExtractionEngine(
                spacy_model=spacy_model,
                cache_size=100  # Smaller cache per language
            )
            self.engines[language] = engine
            return engine

        except Exception as e:
            logger.error(f"Failed to load model for '{language}': {e}")
            # Fallback to default language
            if language != self.default_language:
                logger.info(f"Falling back to default language: {self.default_language}")
                return self._get_or_create_engine(self.default_language)
            raise

    def detect_language(self, text: str) -> str:
        """
        Detect language of text.

        Args:
            text: Input text

        Returns:
            Language code (e.g., 'en', 'es')
        """
        if not self.auto_detect or not self.detector:
            return self.default_language

        # Detect
        results = self.detector.detect(text, top_n=1)

        if results and results[0]['confidence'] > 0.3:
            detected_lang = results[0]['language']

            # Update stats
            if detected_lang not in self.stats['languages_detected']:
                self.stats['languages_detected'][detected_lang] = 0
            self.stats['languages_detected'][detected_lang] += 1

            return detected_lang

        # Fallback to default
        return self.default_language

    def extract(
        self,
        text: str,
        language: Optional[str] = None,
        top_k: int = 10,
        min_confidence: float = 0.3,
        include_entities: bool = True,
        include_concepts: bool = True
    ) -> List[Keyword]:
        """
        Extract keywords with automatic language detection.

        Args:
            text: Input text
            language: Force specific language (None = auto-detect)
            top_k: Number of keywords to return
            min_confidence: Minimum confidence threshold
            include_entities: Include named entities
            include_concepts: Include concept keywords

        Returns:
            List of extracted keywords
        """
        self.stats['total_extractions'] += 1

        # Detect or use specified language
        if language is None:
            language = self.detect_language(text)
        else:
            # Validate language
            if language not in SPACY_MODELS:
                logger.warning(f"Unsupported language '{language}', using default")
                language = self.default_language

        logger.info(f"Extracting keywords (language={language}, top_k={top_k})")

        # Get engine for language
        engine = self._get_or_create_engine(language)

        # Extract
        keywords = engine.extract(
            text=text,
            top_k=top_k,
            min_confidence=min_confidence,
            include_entities=include_entities,
            include_concepts=include_concepts
        )

        logger.info(f"Extracted {len(keywords)} keywords for language '{language}'")

        return keywords

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.

        Returns:
            List of ISO 639-1 language codes
        """
        return list(SPACY_MODELS.keys())

    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics.

        Returns:
            Dict with usage statistics
        """
        return {
            'total_extractions': self.stats['total_extractions'],
            'languages_detected': self.stats['languages_detected'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': (
                self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
                if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0
                else 0.0
            ),
            'loaded_languages': list(self.engines.keys()),
            'supported_languages': self.get_supported_languages(),
            'auto_detect_enabled': self.auto_detect,
            'default_language': self.default_language
        }

    def clear_cache(self):
        """Clear all cached engines."""
        for engine in self.engines.values():
            engine.clear_cache()
        self.engines.clear()
        logger.info("All language engine caches cleared")

    def preload_languages(self, languages: List[str]):
        """
        Preload engines for specific languages.

        Useful for warming up the cache before serving requests.

        Args:
            languages: List of language codes to preload
        """
        logger.info(f"Preloading engines for languages: {languages}")

        for lang in languages:
            if lang not in SPACY_MODELS:
                logger.warning(f"Cannot preload unsupported language: {lang}")
                continue

            try:
                engine = self._get_or_create_engine(lang)
                # Test extraction to ensure model is loaded
                engine.extract("test", top_k=1)
                logger.info(f"Successfully preloaded language: {lang}")

            except Exception as e:
                logger.error(f"Failed to preload language '{lang}': {e}")

        logger.info(f"Preloading complete: {len(self.engines)} languages loaded")
