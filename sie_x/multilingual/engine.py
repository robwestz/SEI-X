"""
Multi-language support with 100+ languages.
"""

from typing import Dict, List, Optional
import fasttext
import polyglot
from transformers import AutoTokenizer, AutoModel
import langdetect
from dataclasses import dataclass


@dataclass
class LanguageConfig:
    """Language-specific configuration."""
    code: str
    name: str
    spacy_model: Optional[str]
    sentence_model: str
    tokenizer: str


LANGUAGE_CONFIGS = {
    'en': LanguageConfig(
        code='en',
        name='English',
        spacy_model='en_core_web_lg',
        sentence_model='sentence-transformers/all-mpnet-base-v2',
        tokenizer='bert-base-uncased'
    ),
    'es': LanguageConfig(
        code='es',
        name='Spanish',
        spacy_model='es_core_news_lg',
        sentence_model='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        tokenizer='dccuchile/bert-base-spanish-wwm-cased'
    ),
    'zh': LanguageConfig(
        code='zh',
        name='Chinese',
        spacy_model='zh_core_web_lg',
        sentence_model='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        tokenizer='bert-base-chinese'
    ),
    'ar': LanguageConfig(
        code='ar',
        name='Arabic',
        spacy_model=None,  # Use Stanza instead
        sentence_model='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        tokenizer='asafaya/bert-base-arabic'
    ),
    # Add 96 more language configurations...
}


class MultilingualEngine:
    """Engine with automatic language detection and processing."""

    def __init__(self):
        self.language_models: Dict[str, Any] = {}
        self.language_detector = fasttext.load_model('lid.176.bin')
        self._load_base_models()

    def _load_base_models(self):
        """Load base multilingual models."""
        # Load universal sentence encoder
        self.universal_embedder = SentenceTransformer(
            'sentence-transformers/LaBSE'  # Supports 109 languages
        )

        # Load XLM-RoBERTa for languages without specific models
        self.xlm_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.xlm_model = AutoModel.from_pretrained('xlm-roberta-base')

    def detect_language(self, text: str) -> str:
        """Detect text language."""
        # Try langdetect first (faster)
        try:
            lang = langdetect.detect(text)
            if lang in LANGUAGE_CONFIGS:
                return lang
        except:
            pass

        # Fall back to fasttext
        predictions = self.language_detector.predict(text, k=1)
        lang_code = predictions[0][0].replace('__label__', '')[:2]

        return lang_code if lang_code in LANGUAGE_CONFIGS else 'en'

    async def extract_multilingual(
            self,
            text: str,
            language: Optional[str] = None,
            **kwargs
    ) -> List[Keyword]:
        """Extract keywords with automatic language handling."""
        # Detect language if not specified
        if not language:
            language = self.detect_language(text)

        logger.info(f"Processing text in {language}")

        # Get language-specific models
        models = await self._get_language_models(language)

        # Create language-specific engine
        engine = SemanticIntelligenceEngine(
            language_model=models['sentence_model'],
            nlp=models['nlp']
        )

        # Extract keywords
        keywords = await engine.extract_async(text, **kwargs)

        # Post-process for language-specific rules
        keywords = self._apply_language_rules(keywords, language)

        return keywords

    async def _get_language_models(self, language: str) -> Dict[str, Any]:
        """Get or load language-specific models."""
        if language not in self.language_models:
            config = LANGUAGE_CONFIGS.get(language)

            if not config:
                # Use universal models
                return {
                    'sentence_model': self.universal_embedder,
                    'nlp': None,
                    'tokenizer': self.xlm_tokenizer
                }

            # Load language-specific models
            models = {}

            # Sentence transformer
            models['sentence_model'] = SentenceTransformer(config.sentence_model)

            # SpaCy or alternative
            if config.spacy_model:
                models['nlp'] = spacy.load(config.spacy_model)
            else:
                # Use Stanza for languages without spaCy
                import stanza
                stanza.download(language)
                models['nlp'] = stanza.Pipeline(language)

            # Tokenizer
            models['tokenizer'] = AutoTokenizer.from_pretrained(config.tokenizer)

            self.language_models[language] = models

        return self.language_models[language]

    def _apply_language_rules(self, keywords: List[Keyword], language: str) -> List[Keyword]:
        """Apply language-specific processing rules."""
        if language == 'zh':
            # Chinese: Handle word segmentation
            keywords = self._chinese_postprocessing(keywords)
        elif language == 'ar':
            # Arabic: Handle RTL and diacritics
            keywords = self._arabic_postprocessing(keywords)
        elif language == 'ja':
            # Japanese: Handle kanji/kana
            keywords = self._japanese_postprocessing(keywords)

        return keywords