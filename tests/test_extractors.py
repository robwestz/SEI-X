"""Tests for CandidateExtractor, TermFilter and helper functions."""

import pytest

from sie_x.core.extractors import (
    CandidateExtractor,
    TermFilter,
    merge_overlapping_phrases,
    deduplicate_phrases,
)


# ============================================================================
# TermFilter.is_valid
# ============================================================================

class TestTermFilterIsValid:

    def setup_method(self):
        self.tf = TermFilter()

    def test_valid_term(self):
        assert self.tf.is_valid("machine learning") is True

    def test_too_short(self):
        assert self.tf.is_valid("x") is False

    def test_too_long(self):
        assert self.tf.is_valid("a" * 51) is False

    def test_only_numbers(self):
        assert self.tf.is_valid("12345") is False

    def test_only_punctuation(self):
        assert self.tf.is_valid("!!!") is False

    def test_url_rejected(self):
        assert self.tf.is_valid("http://example.com") is False
        assert self.tf.is_valid("www.example.com") is False

    def test_social_handle_rejected(self):
        assert self.tf.is_valid("@user123") is False
        assert self.tf.is_valid("#hashtag") is False

    def test_noise_term_rejected(self):
        assert self.tf.is_valid("click here") is False
        assert self.tf.is_valid("read more") is False

    def test_mostly_punctuation_rejected(self):
        assert self.tf.is_valid("---!@#") is False

    def test_custom_stop_pattern(self):
        tf = TermFilter(custom_stop_patterns=[r'^ignore_me$'])
        assert tf.is_valid("ignore_me") is False
        assert tf.is_valid("keep_me") is True


# ============================================================================
# TermFilter.filter_candidates / filter_by_frequency
# ============================================================================

class TestTermFilterBatch:

    def test_filter_candidates(self):
        tf = TermFilter()
        candidates = ["machine learning", "x", "12345", "Python"]
        result = tf.filter_candidates(candidates)
        assert "machine learning" in result
        assert "Python" in result
        assert "x" not in result
        assert "12345" not in result

    def test_filter_by_frequency_min(self):
        tf = TermFilter()
        candidates = ["ai", "ai", "ai", "ml", "ml", "data"]
        result = tf.filter_by_frequency(candidates, min_frequency=2)
        assert "ai" in result
        assert "ml" in result
        assert "data" not in result

    def test_filter_by_frequency_max(self):
        tf = TermFilter()
        candidates = ["ai", "ai", "ai", "ml"]
        result = tf.filter_by_frequency(candidates, min_frequency=1, max_frequency=2)
        assert "ml" in result
        assert "ai" not in result


# ============================================================================
# merge_overlapping_phrases
# ============================================================================

class TestMergeOverlapping:

    def test_empty_list(self):
        assert merge_overlapping_phrases([]) == []

    def test_no_overlap(self):
        phrases = [("machine learning", 0.9), ("data science", 0.8)]
        result = merge_overlapping_phrases(phrases)
        assert len(result) == 2

    def test_overlap_keeps_higher_score(self):
        phrases = [
            ("machine learning", 0.9),
            ("learning", 0.5),
        ]
        result = merge_overlapping_phrases(phrases)
        texts = [p[0] for p in result]
        assert "machine learning" in texts
        # "learning" has 100% overlap with "machine learning" words
        assert "learning" not in texts


# ============================================================================
# deduplicate_phrases
# ============================================================================

class TestDeduplicatePhrases:

    def test_no_duplicates(self):
        result = deduplicate_phrases(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_case_insensitive(self):
        result = deduplicate_phrases(["Python", "python", "PYTHON"])
        assert result == ["Python"]

    def test_preserves_order(self):
        result = deduplicate_phrases(["beta", "alpha", "beta"])
        assert result == ["beta", "alpha"]
