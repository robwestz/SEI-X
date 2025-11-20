"""
SIE-X Streaming Support

Enables processing of large documents by chunking and streaming results.
Useful for:
- Long articles and books (>10K words)
- Real-time extraction feedback
- Memory-efficient processing
- Progressive UI updates

Example:
    >>> from sie_x.core.streaming import StreamingExtractor
    >>>
    >>> extractor = StreamingExtractor()
    >>> async for chunk_result in extractor.extract_stream(long_text):
    ...     print(f"Chunk {chunk_result['chunk_id']}: {len(chunk_result['keywords'])} keywords")
"""

from typing import AsyncGenerator, Dict, List, Any, Optional
import asyncio
from dataclasses import dataclass
import logging

from sie_x.core.models import Keyword, ExtractionOptions
from sie_x.core.simple_engine import SimpleExtractionEngine

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    chunk_size: int = 1000  # Words per chunk
    overlap: int = 100      # Overlap between chunks (words)
    min_chunk_size: int = 200  # Minimum chunk size to process
    strategy: str = "smart"  # 'simple' (word-based) or 'smart' (structure-aware)

    def validate(self):
        """Validate chunk configuration."""
        if self.chunk_size < self.min_chunk_size:
            raise ValueError(f"chunk_size ({self.chunk_size}) must be >= min_chunk_size ({self.min_chunk_size})")
        if self.overlap >= self.chunk_size:
            raise ValueError(f"overlap ({self.overlap}) must be < chunk_size ({self.chunk_size})")
        if self.overlap < 0:
            raise ValueError(f"overlap ({self.overlap}) must be >= 0")
        if self.strategy not in ('simple', 'smart'):
            raise ValueError(f"strategy must be 'simple' or 'smart', got '{self.strategy}'")


class StreamingExtractor:
    """
    Streaming keyword extractor for large documents.

    Processes text in configurable chunks and yields results progressively.
    Useful for documents >10K words or when memory is constrained.

    Attributes:
        engine: Core extraction engine
        chunk_config: Chunking configuration
        merge_strategy: How to merge keywords across chunks ('union', 'intersection', 'weighted')
    """

    def __init__(
        self,
        engine: Optional[SimpleExtractionEngine] = None,
        chunk_config: Optional[ChunkConfig] = None,
        merge_strategy: str = "weighted"
    ):
        """
        Initialize streaming extractor.

        Args:
            engine: Extraction engine (creates new one if None)
            chunk_config: Chunking configuration
            merge_strategy: 'union' (all keywords), 'intersection' (common only), 'weighted' (frequency-weighted)
        """
        self.engine = engine or SimpleExtractionEngine()
        self.chunk_config = chunk_config or ChunkConfig()
        self.chunk_config.validate()
        self.merge_strategy = merge_strategy

        logger.info(f"StreamingExtractor initialized: chunk_size={self.chunk_config.chunk_size}, "
                   f"overlap={self.chunk_config.overlap}, merge={merge_strategy}")

    def _split_into_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.

        Uses configured strategy: 'simple' (word-based) or 'smart' (structure-aware).

        Args:
            text: Input text

        Returns:
            List of chunk dicts with id, text, start_word, end_word
        """
        if self.chunk_config.strategy == "smart":
            return self._smart_chunk(text)
        else:
            return self._simple_chunk(text)

    def _simple_chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        Simple word-based chunking (original implementation).

        Args:
            text: Input text

        Returns:
            List of chunk dicts
        """
        words = text.split()
        total_words = len(words)

        if total_words <= self.chunk_config.chunk_size:
            # Text is small enough, return as single chunk
            return [{
                'chunk_id': 0,
                'text': text,
                'start_word': 0,
                'end_word': total_words,
                'is_final': True,
                'total_chunks': 1
            }]

        chunks = []
        chunk_id = 0
        start = 0

        while start < total_words:
            end = min(start + self.chunk_config.chunk_size, total_words)

            # Extract chunk words
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)

            # Skip if chunk too small (unless it's the last one)
            if len(chunk_words) < self.chunk_config.min_chunk_size and end < total_words:
                start = end - self.chunk_config.overlap
                continue

            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'start_word': start,
                'end_word': end,
                'is_final': False,
                'total_chunks': -1  # Unknown until we finish
            })

            chunk_id += 1
            start = end - self.chunk_config.overlap

        # Mark last chunk and set total
        if chunks:
            chunks[-1]['is_final'] = True
            for chunk in chunks:
                chunk['total_chunks'] = len(chunks)

        logger.info(f"Split text into {len(chunks)} chunks ({total_words} words)")

        return chunks

    def _smart_chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        Smart structure-aware chunking (respects paragraphs, headers, sentences).

        Strategy:
        1. Split on paragraph boundaries (\n\n or \n for markdown)
        2. Detect headers (lines starting with #)
        3. Group paragraphs into chunks while respecting target size
        4. Maintain overlap for context continuity
        5. Fall back to sentence/word splitting for very long paragraphs

        Args:
            text: Input text

        Returns:
            List of chunk dicts
        """
        # Split into paragraphs (double newline or single newline for markdown)
        paragraphs = []
        current_para = []

        for line in text.split('\n'):
            line_stripped = line.strip()

            # Empty line = paragraph boundary
            if not line_stripped:
                if current_para:
                    paragraphs.append('\n'.join(current_para))
                    current_para = []
            # Markdown header = new paragraph
            elif line_stripped.startswith('#'):
                if current_para:
                    paragraphs.append('\n'.join(current_para))
                    current_para = []
                paragraphs.append(line)  # Header as standalone paragraph
            else:
                current_para.append(line)

        # Add final paragraph
        if current_para:
            paragraphs.append('\n'.join(current_para))

        # If no clear paragraph structure, fall back to simple chunking
        if len(paragraphs) <= 1:
            logger.info("No clear paragraph structure detected, using simple chunking")
            return self._simple_chunk(text)

        # Count words in each paragraph
        para_word_counts = [len(p.split()) for p in paragraphs]
        total_words = sum(para_word_counts)

        # If document is small, return as single chunk
        if total_words <= self.chunk_config.chunk_size:
            return [{
                'chunk_id': 0,
                'text': text,
                'start_word': 0,
                'end_word': total_words,
                'is_final': True,
                'total_chunks': 1
            }]

        # Group paragraphs into chunks
        chunks = []
        chunk_id = 0
        current_chunk_paras = []
        current_chunk_words = 0
        word_position = 0
        chunk_start_word = 0

        # For overlap: keep last N paragraphs from previous chunk
        overlap_paras = []
        overlap_words = 0

        for i, (para, para_words) in enumerate(zip(paragraphs, para_word_counts)):
            # If single paragraph exceeds chunk size, split it
            if para_words > self.chunk_config.chunk_size * 1.5:
                # Flush current chunk if any
                if current_chunk_paras:
                    chunk_text = '\n\n'.join(current_chunk_paras)
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': chunk_text,
                        'start_word': chunk_start_word,
                        'end_word': word_position,
                        'is_final': False,
                        'total_chunks': -1
                    })
                    chunk_id += 1

                    # Calculate overlap for next chunk
                    overlap_paras = self._get_overlap_paragraphs(current_chunk_paras, para_word_counts)
                    overlap_words = sum(len(p.split()) for p in overlap_paras)

                # Split the long paragraph by sentences
                long_para_chunks = self._split_long_paragraph(para, para_words, word_position)
                chunks.extend(long_para_chunks)
                chunk_id += len(long_para_chunks)

                word_position += para_words
                current_chunk_paras = []
                current_chunk_words = 0
                chunk_start_word = word_position
                overlap_paras = []
                overlap_words = 0

            # Check if adding this paragraph would exceed chunk size
            elif current_chunk_words + para_words > self.chunk_config.chunk_size:
                # Create chunk from accumulated paragraphs
                if current_chunk_paras:
                    chunk_text = '\n\n'.join(current_chunk_paras)
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': chunk_text,
                        'start_word': chunk_start_word,
                        'end_word': word_position,
                        'is_final': False,
                        'total_chunks': -1
                    })
                    chunk_id += 1

                    # Calculate overlap for next chunk
                    overlap_paras = self._get_overlap_paragraphs(current_chunk_paras, para_word_counts)
                    overlap_words = sum(len(p.split()) for p in overlap_paras)

                # Start new chunk with overlap + current paragraph
                current_chunk_paras = overlap_paras + [para]
                chunk_start_word = word_position - overlap_words
                current_chunk_words = overlap_words + para_words
                word_position += para_words

            else:
                # Add paragraph to current chunk
                current_chunk_paras.append(para)
                current_chunk_words += para_words
                word_position += para_words

        # Add final chunk if any paragraphs remain
        if current_chunk_paras:
            chunk_text = '\n\n'.join(current_chunk_paras)
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'start_word': chunk_start_word,
                'end_word': word_position,
                'is_final': False,
                'total_chunks': -1
            })

        # Mark last chunk and set total
        if chunks:
            chunks[-1]['is_final'] = True
            for chunk in chunks:
                chunk['total_chunks'] = len(chunks)

        logger.info(f"Smart chunking: {len(chunks)} chunks from {len(paragraphs)} paragraphs ({total_words} words)")

        return chunks

    def _get_overlap_paragraphs(
        self,
        paragraphs: List[str],
        para_word_counts: List[int]
    ) -> List[str]:
        """
        Get last N paragraphs from previous chunk for overlap.

        Args:
            paragraphs: List of paragraph strings
            para_word_counts: Word counts for each paragraph

        Returns:
            List of paragraphs for overlap
        """
        overlap_target = self.chunk_config.overlap
        overlap_paras = []
        overlap_words = 0

        # Work backwards from end of paragraphs
        for para in reversed(paragraphs):
            para_words = len(para.split())
            if overlap_words + para_words <= overlap_target:
                overlap_paras.insert(0, para)
                overlap_words += para_words
            else:
                # If we can't fit the whole paragraph, take last N words
                if overlap_words < overlap_target:
                    words_needed = overlap_target - overlap_words
                    para_words_list = para.split()
                    if len(para_words_list) > words_needed:
                        partial = ' '.join(para_words_list[-words_needed:])
                        overlap_paras.insert(0, partial)
                break

        return overlap_paras

    def _split_long_paragraph(
        self,
        paragraph: str,
        para_words: int,
        start_word_pos: int
    ) -> List[Dict[str, Any]]:
        """
        Split a very long paragraph into sentence-based or word-based chunks.

        Args:
            paragraph: The long paragraph text
            para_words: Word count of paragraph
            start_word_pos: Starting word position in document

        Returns:
            List of chunk dicts for this paragraph
        """
        # Try sentence splitting first
        import re
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)

        if len(sentences) > 1:
            # Use sentence-based chunking
            chunks = []
            current_chunk_sentences = []
            current_chunk_words = 0
            word_pos = start_word_pos
            chunk_start = start_word_pos

            for sentence in sentences:
                sentence_words = len(sentence.split())

                if current_chunk_words + sentence_words > self.chunk_config.chunk_size:
                    if current_chunk_sentences:
                        chunk_text = ' '.join(current_chunk_sentences)
                        chunks.append({
                            'chunk_id': -1,  # Will be renumbered by caller
                            'text': chunk_text,
                            'start_word': chunk_start,
                            'end_word': word_pos,
                            'is_final': False,
                            'total_chunks': -1
                        })

                    current_chunk_sentences = [sentence]
                    current_chunk_words = sentence_words
                    chunk_start = word_pos
                else:
                    current_chunk_sentences.append(sentence)
                    current_chunk_words += sentence_words

                word_pos += sentence_words

            # Add final sentence chunk
            if current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append({
                    'chunk_id': -1,
                    'text': chunk_text,
                    'start_word': chunk_start,
                    'end_word': word_pos,
                    'is_final': False,
                    'total_chunks': -1
                })

            return chunks

        else:
            # No sentence structure, fall back to word splitting
            words = paragraph.split()
            chunks = []
            start = 0
            word_pos = start_word_pos

            while start < len(words):
                end = min(start + self.chunk_config.chunk_size, len(words))
                chunk_words = words[start:end]
                chunk_text = ' '.join(chunk_words)

                chunks.append({
                    'chunk_id': -1,
                    'text': chunk_text,
                    'start_word': word_pos + start,
                    'end_word': word_pos + end,
                    'is_final': False,
                    'total_chunks': -1
                })

                start = end - self.chunk_config.overlap

            return chunks

    async def extract_stream(
        self,
        text: str,
        top_k: int = 10,
        min_confidence: float = 0.3,
        merge_final: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream keyword extraction results chunk by chunk.

        Args:
            text: Input text
            top_k: Keywords per chunk
            min_confidence: Minimum confidence threshold
            merge_final: If True, yields final merged result as last item

        Yields:
            Dict with:
            - chunk_id: Chunk identifier
            - keywords: List[Keyword] for this chunk
            - is_final_chunk: Whether this is the last chunk
            - is_merged_result: Whether this is the final merged result
            - progress: Percentage complete (0-100)
            - metadata: Chunk metadata
        """
        chunks = self._split_into_chunks(text)
        all_keywords: List[List[Keyword]] = []

        for i, chunk_info in enumerate(chunks):
            # Extract keywords for this chunk
            chunk_keywords = self.engine.extract(
                text=chunk_info['text'],
                top_k=top_k,
                min_confidence=min_confidence
            )

            all_keywords.append(chunk_keywords)

            # Calculate progress
            progress = ((i + 1) / len(chunks)) * 100

            # Yield chunk result
            yield {
                'chunk_id': chunk_info['chunk_id'],
                'keywords': [kw.model_dump() for kw in chunk_keywords],
                'is_final_chunk': chunk_info['is_final'],
                'is_merged_result': False,
                'progress': round(progress, 1),
                'metadata': {
                    'start_word': chunk_info['start_word'],
                    'end_word': chunk_info['end_word'],
                    'total_chunks': chunk_info['total_chunks']
                }
            }

            # Small delay to avoid overwhelming consumers
            await asyncio.sleep(0.01)

        # Yield final merged result
        if merge_final and all_keywords:
            merged = self._merge_keywords(all_keywords, top_k)

            yield {
                'chunk_id': -1,
                'keywords': [kw.model_dump() for kw in merged],
                'is_final_chunk': True,
                'is_merged_result': True,
                'progress': 100.0,
                'metadata': {
                    'total_chunks': len(chunks),
                    'merge_strategy': self.merge_strategy
                }
            }

    def _merge_keywords(
        self,
        keyword_lists: List[List[Keyword]],
        top_k: int
    ) -> List[Keyword]:
        """
        Merge keywords from multiple chunks.

        Args:
            keyword_lists: List of keyword lists from each chunk
            top_k: Number of final keywords to return

        Returns:
            Merged and ranked keywords
        """
        if self.merge_strategy == "union":
            return self._merge_union(keyword_lists, top_k)
        elif self.merge_strategy == "intersection":
            return self._merge_intersection(keyword_lists, top_k)
        elif self.merge_strategy == "weighted":
            return self._merge_weighted(keyword_lists, top_k)
        else:
            logger.warning(f"Unknown merge strategy '{self.merge_strategy}', using weighted")
            return self._merge_weighted(keyword_lists, top_k)

    def _merge_union(self, keyword_lists: List[List[Keyword]], top_k: int) -> List[Keyword]:
        """Merge using union - all unique keywords."""
        keyword_map: Dict[str, Keyword] = {}

        for kw_list in keyword_lists:
            for kw in kw_list:
                key = kw.text.lower()
                if key not in keyword_map:
                    keyword_map[key] = kw
                else:
                    # Take higher score
                    if kw.score > keyword_map[key].score:
                        keyword_map[key] = kw

        # Sort by score and return top_k
        merged = sorted(keyword_map.values(), key=lambda x: x.score, reverse=True)
        return merged[:top_k]

    def _merge_intersection(self, keyword_lists: List[List[Keyword]], top_k: int) -> List[Keyword]:
        """Merge using intersection - only keywords in all chunks."""
        if not keyword_lists:
            return []

        # Get keywords that appear in all chunks
        keyword_sets = [set(kw.text.lower() for kw in kw_list) for kw_list in keyword_lists]
        common = keyword_sets[0].intersection(*keyword_sets[1:])

        # Collect keywords and average scores
        keyword_map: Dict[str, List[Keyword]] = {key: [] for key in common}

        for kw_list in keyword_lists:
            for kw in kw_list:
                key = kw.text.lower()
                if key in common:
                    keyword_map[key].append(kw)

        # Average scores
        merged = []
        for key, kws in keyword_map.items():
            avg_score = sum(kw.score for kw in kws) / len(kws)
            avg_confidence = sum(kw.confidence for kw in kws) / len(kws)
            total_count = sum(kw.count for kw in kws)

            merged_kw = Keyword(
                text=kws[0].text,
                score=avg_score,
                type=kws[0].type,
                count=total_count,
                confidence=avg_confidence
            )
            merged.append(merged_kw)

        # Sort by score
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:top_k]

    def _merge_weighted(self, keyword_lists: List[List[Keyword]], top_k: int) -> List[Keyword]:
        """Merge using weighted frequency - keywords weighted by appearance count."""
        keyword_map: Dict[str, List[Keyword]] = {}

        for kw_list in keyword_lists:
            for kw in kw_list:
                key = kw.text.lower()
                if key not in keyword_map:
                    keyword_map[key] = []
                keyword_map[key].append(kw)

        # Calculate weighted scores
        merged = []
        for key, kws in keyword_map.items():
            # Weight by frequency across chunks
            frequency_weight = len(kws) / len(keyword_lists)
            avg_score = sum(kw.score for kw in kws) / len(kws)
            weighted_score = avg_score * (0.7 + 0.3 * frequency_weight)  # Boost frequent keywords

            avg_confidence = sum(kw.confidence for kw in kws) / len(kws)
            total_count = sum(kw.count for kw in kws)

            merged_kw = Keyword(
                text=kws[0].text,
                score=min(weighted_score, 1.0),  # Cap at 1.0
                type=kws[0].type,
                count=total_count,
                confidence=avg_confidence
            )
            merged.append(merged_kw)

        # Sort by weighted score
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:top_k]

    def extract_sync(
        self,
        text: str,
        top_k: int = 10,
        min_confidence: float = 0.3
    ) -> Dict[str, Any]:
        """
        Synchronous extraction with chunking (no streaming).

        Processes in chunks but returns final merged result only.
        Useful when streaming is not needed but chunking is (memory limits).

        Args:
            text: Input text
            top_k: Number of keywords to return
            min_confidence: Minimum confidence threshold

        Returns:
            Dict with merged keywords and metadata
        """
        chunks = self._split_into_chunks(text)
        all_keywords: List[List[Keyword]] = []

        logger.info(f"Processing {len(chunks)} chunks synchronously")

        for chunk_info in chunks:
            chunk_keywords = self.engine.extract(
                text=chunk_info['text'],
                top_k=top_k,
                min_confidence=min_confidence
            )
            all_keywords.append(chunk_keywords)

        # Merge all results
        merged = self._merge_keywords(all_keywords, top_k)

        return {
            'keywords': [kw.model_dump() for kw in merged],
            'metadata': {
                'total_chunks': len(chunks),
                'merge_strategy': self.merge_strategy,
                'total_words': sum(c['end_word'] - c['start_word'] for c in chunks)
            }
        }
