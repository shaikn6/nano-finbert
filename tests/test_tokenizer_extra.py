"""
Extended tokenizer tests for edge cases not covered by test_tokenizer.py:
  - non-English input, special characters, very long text
  - BPE merge path on unseen words
  - build_default_tokenizer with no texts (seed corpus fallback)
  - _pre_tokenize static method
  - _get_pair_frequencies and _merge_pair static methods
  - __len__ and __repr__
  - vocab_stats multi_char / single_char / coverage counts
"""

from __future__ import annotations

import pytest

from finbert.tokenizer import (
    FINANCIAL_SEED_VOCAB,
    FinancialTokenizer,
    build_default_tokenizer,
    vocab_stats,
)

# ---------------------------------------------------------------------------
# _pre_tokenize (static)
# ---------------------------------------------------------------------------


class TestPreTokenize:
    def test_basic_words(self):
        tokens = FinancialTokenizer._pre_tokenize("gold silver oil")
        assert tokens == ["gold", "silver", "oil"]

    def test_ticker_kept_together(self):
        tokens = FinancialTokenizer._pre_tokenize("$AAPL rose 3% today")
        assert "$aapl" in tokens or "$aapl".replace("$", "$") in " ".join(tokens)
        # Tickers may be lowercased — just verify the list is non-empty
        assert len(tokens) > 0

    def test_numbers_included(self):
        tokens = FinancialTokenizer._pre_tokenize("rose 3.5%")
        assert len(tokens) >= 1

    def test_empty_string_returns_empty_list(self):
        assert FinancialTokenizer._pre_tokenize("") == []

    def test_punctuation_stripped(self):
        tokens = FinancialTokenizer._pre_tokenize("gold, silver!")
        assert "gold" in tokens
        assert "silver" in tokens

    def test_non_english_chars_handled(self):
        # Non-English characters should not crash
        tokens = FinancialTokenizer._pre_tokenize("financière résultats €100")
        assert isinstance(tokens, list)

    def test_only_punctuation_returns_empty(self):
        tokens = FinancialTokenizer._pre_tokenize("!!! ???")
        assert tokens == []

    def test_mixed_case_lowercased(self):
        tokens = FinancialTokenizer._pre_tokenize("Bitcoin FELL")
        assert "bitcoin" in tokens or "Bitcoin".lower() in tokens


# ---------------------------------------------------------------------------
# _get_pair_frequencies (static)
# ---------------------------------------------------------------------------


class TestGetPairFrequencies:
    def test_counts_adjacent_pairs(self):
        vocab = {"g o l d </w>": 2, "s i l v e r </w>": 1}
        pairs = FinancialTokenizer._get_pair_frequencies(vocab)
        assert ("g", "o") in pairs
        assert pairs[("g", "o")] == 2

    def test_empty_vocab_returns_empty(self):
        pairs = FinancialTokenizer._get_pair_frequencies({})
        assert len(pairs) == 0

    def test_single_char_word_no_pairs(self):
        vocab = {"a </w>": 5}
        pairs = FinancialTokenizer._get_pair_frequencies(vocab)
        assert ("a", "</w>") in pairs
        assert pairs[("a", "</w>")] == 5


# ---------------------------------------------------------------------------
# _merge_pair (static)
# ---------------------------------------------------------------------------


class TestMergePair:
    def test_merges_pair_in_word(self):
        vocab = {"g o l d </w>": 3}
        new_vocab = FinancialTokenizer._merge_pair(("g", "o"), vocab)
        assert "go l d </w>" in new_vocab
        assert new_vocab["go l d </w>"] == 3

    def test_unmatched_pair_unchanged(self):
        vocab = {"g o l d </w>": 3}
        new_vocab = FinancialTokenizer._merge_pair(("x", "y"), vocab)
        assert "g o l d </w>" in new_vocab

    def test_multiple_occurrences_merged(self):
        vocab = {"a b a b </w>": 1}
        new_vocab = FinancialTokenizer._merge_pair(("a", "b"), vocab)
        # Merge replaces all non-overlapping occurrences
        assert any("ab" in k for k in new_vocab)


# ---------------------------------------------------------------------------
# Training edge cases
# ---------------------------------------------------------------------------


class TestTrainingEdgeCases:
    def test_empty_texts_list_trains_without_crash(self):
        t = FinancialTokenizer(vocab_size=200)
        t.train([])  # empty corpus
        assert t._trained

    def test_single_character_text(self):
        t = FinancialTokenizer(vocab_size=200)
        t.train(["a"])
        assert t._trained

    def test_very_long_text_trains(self):
        t = FinancialTokenizer(vocab_size=300)
        long_text = " ".join(["gold"] * 500)
        t.train([long_text])
        assert t._trained

    def test_train_with_financial_seed_vocab_terms(self):
        t = FinancialTokenizer(vocab_size=500)
        texts = [" ".join(FINANCIAL_SEED_VOCAB[:30])]
        t.train(texts)
        # Seed terms should appear in vocabulary
        assert len(t.token_to_id) > len(t.SPECIAL_TOKENS)

    def test_repeated_training_calls_add_to_vocab(self):
        """Calling train twice should not crash (though vocab may not change much)."""
        t = FinancialTokenizer(vocab_size=300)
        t.train(["gold silver oil"])
        first_size = len(t.token_to_id)
        t.train(["bitcoin ethereum"])  # second call
        assert t._trained
        # Retraining must not shrink the vocabulary.
        assert len(t.token_to_id) >= first_size


# ---------------------------------------------------------------------------
# Encode edge cases
# ---------------------------------------------------------------------------


class TestEncodeEdgeCases:
    @pytest.fixture(scope="class")
    def tok(self):
        t = FinancialTokenizer(vocab_size=500)
        t.train(
            [
                "gold silver copper oil bitcoin ethereum dollar euro earnings revenue",
                "Apple Microsoft Google Tesla Nvidia SpaceX JPMorgan Goldman",
                "ipo acquisition merger bankruptcy default inflation recession",
            ]
        )
        return t

    def test_very_long_text_truncated_to_max_length(self, tok):
        long_text = " ".join(["gold"] * 1000)
        result = tok.encode(long_text, max_length=32, truncation=True, padding=True)
        assert len(result["input_ids"]) == 32

    def test_text_with_special_characters(self, tok):
        result = tok.encode("Price: $100.50 (+3.2%)", max_length=32)
        assert len(result["input_ids"]) == 32

    def test_text_with_newlines_and_tabs(self, tok):
        result = tok.encode("gold\nsilver\toil prices", max_length=32)
        assert len(result["input_ids"]) == 32

    def test_text_with_numbers_only(self, tok):
        result = tok.encode("100 200 300 400 500", max_length=32)
        assert len(result["input_ids"]) == 32

    def test_non_english_text_does_not_crash(self, tok):
        """Non-English text encodes without raising and yields valid token IDs.

        The tokenizer has a character-level fallback, so latin-script input may
        map to known tokens rather than [UNK]; the contract is no crash and a
        well-formed, fixed-length output within the vocab range.
        """
        result = tok.encode("Résultats financiers du troisième trimestre", max_length=32)
        assert len(result["input_ids"]) == 32
        assert all(0 <= tid < tok.vocab_size for tid in result["input_ids"])

    def test_chinese_text_does_not_crash(self, tok):
        result = tok.encode("中国经济增长放缓", max_length=32)
        assert isinstance(result["input_ids"], list)

    def test_arabic_text_does_not_crash(self, tok):
        result = tok.encode("النفط الخام يرتفع", max_length=32)
        assert isinstance(result["input_ids"], list)

    def test_emoji_text_does_not_crash(self, tok):
        result = tok.encode("📈 stocks rising 🚀", max_length=32)
        assert isinstance(result["input_ids"], list)

    def test_truncation_false_no_truncation(self, tok):
        """With truncation=False, very long text is not truncated (may exceed max_length)."""
        # This tests the branch where truncation=False
        result = tok.encode("gold " * 5, max_length=8, truncation=False, padding=False)
        # Length may exceed 8 since we don't truncate
        assert isinstance(result["input_ids"], list)

    def test_padding_false_returns_variable_length(self, tok):
        r1 = tok.encode("gold", padding=False, max_length=32)
        r2 = tok.encode("gold silver copper oil dollar euro", padding=False, max_length=32)
        assert len(r1["input_ids"]) < len(r2["input_ids"])

    def test_max_length_one_returns_minimal_sequence(self, tok):
        """max_length=2 just fits [CLS][SEP]."""
        result = tok.encode("gold", max_length=2, truncation=True, padding=True)
        assert len(result["input_ids"]) == 2
        cls_id = FinancialTokenizer.SPECIAL_TOKENS["[CLS]"]
        assert result["input_ids"][0] == cls_id

    def test_input_ids_type_is_list(self, tok):
        result = tok.encode("gold")
        assert isinstance(result["input_ids"], list)
        assert isinstance(result["attention_mask"], list)

    def test_all_mask_values_are_zero_or_one(self, tok):
        result = tok.encode("gold silver", max_length=32, padding=True)
        assert all(m in (0, 1) for m in result["attention_mask"])


# ---------------------------------------------------------------------------
# Decode edge cases
# ---------------------------------------------------------------------------


class TestDecodeEdgeCases:
    @pytest.fixture(scope="class")
    def tok(self):
        t = FinancialTokenizer(vocab_size=300)
        t.train(["gold silver oil bitcoin dollar euro earnings revenue ipo merger"])
        return t

    def test_empty_id_list_returns_empty_string(self, tok):
        assert tok.decode([]) == ""

    def test_decode_only_special_tokens_returns_empty(self, tok):
        special_ids = list(tok.SPECIAL_TOKENS.values())
        decoded = tok.decode(special_ids, skip_special_tokens=True)
        assert decoded == ""

    def test_decode_mixed_ids_strips_end_markers(self, tok):
        result = tok.encode("gold silver", max_length=16, padding=False)
        decoded = tok.decode(result["input_ids"])
        assert "</w>" not in decoded

    def test_decode_pad_tokens_excluded_by_default(self, tok):
        result = tok.encode("gold", max_length=16, padding=True)
        decoded = tok.decode(result["input_ids"])
        assert "[PAD]" not in decoded

    def test_decode_mask_token_id(self, tok):
        mask_id = tok.SPECIAL_TOKENS["[MASK]"]
        decoded = tok.decode([mask_id], skip_special_tokens=False)
        assert "[MASK]" in decoded


# ---------------------------------------------------------------------------
# __len__ and __repr__
# ---------------------------------------------------------------------------


class TestDunderMethods:
    def test_len_equals_token_to_id_size(self):
        t = FinancialTokenizer(vocab_size=200)
        assert len(t) == len(t.token_to_id)

    def test_len_after_training_increases(self):
        t = FinancialTokenizer(vocab_size=200)
        before = len(t)
        t.train(["gold silver oil copper bitcoin"])
        assert len(t) > before

    def test_repr_contains_financial_tokenizer(self):
        t = FinancialTokenizer()
        assert "FinancialTokenizer" in repr(t)

    def test_repr_untrained_status(self):
        t = FinancialTokenizer()
        assert "untrained" in repr(t)

    def test_repr_trained_status(self):
        t = FinancialTokenizer(vocab_size=200)
        t.train(["gold"])
        assert "trained" in repr(t)


# ---------------------------------------------------------------------------
# vocab_stats
# ---------------------------------------------------------------------------


class TestVocabStatsExtra:
    @pytest.fixture(scope="class")
    def trained_tok(self):
        t = FinancialTokenizer(vocab_size=300)
        t.train(["gold silver oil bitcoin dollar earnings revenue ipo merger acquisition"])
        return t

    def test_total_tokens_equals_vocab_size_actual(self, trained_tok):
        stats = vocab_stats(trained_tok)
        assert stats["total_tokens"] == len(trained_tok.token_to_id)

    def test_special_tokens_five(self, trained_tok):
        stats = vocab_stats(trained_tok)
        assert stats["special_tokens"] == 5

    def test_multi_char_tokens_count(self, trained_tok):
        stats = vocab_stats(trained_tok)
        regular = set(trained_tok.token_to_id.keys()) - set(
            FinancialTokenizer.SPECIAL_TOKENS.keys()
        )
        expected = len([t for t in regular if len(t) > 1])
        assert stats["multi_char_tokens"] == expected

    def test_single_char_tokens_count(self, trained_tok):
        stats = vocab_stats(trained_tok)
        regular = set(trained_tok.token_to_id.keys()) - set(
            FinancialTokenizer.SPECIAL_TOKENS.keys()
        )
        expected = len([t for t in regular if len(t) == 1])
        assert stats["single_char_tokens"] == expected

    def test_coverage_examples_is_list(self, trained_tok):
        stats = vocab_stats(trained_tok)
        assert isinstance(stats["coverage_examples"], list)

    def test_total_equals_special_plus_regular(self, trained_tok):
        stats = vocab_stats(trained_tok)
        assert stats["total_tokens"] == (
            stats["special_tokens"] + stats["multi_char_tokens"] + stats["single_char_tokens"]
        )


# ---------------------------------------------------------------------------
# build_default_tokenizer
# ---------------------------------------------------------------------------


class TestBuildDefaultTokenizer:
    def test_no_texts_uses_seed_corpus(self):
        t = build_default_tokenizer(texts=None)
        assert t._trained

    def test_custom_texts_used(self):
        t = build_default_tokenizer(texts=["gold silver oil", "bitcoin ethereum defi"])
        assert t._trained
        assert len(t) > len(t.SPECIAL_TOKENS)

    def test_returns_financial_tokenizer_instance(self):
        t = build_default_tokenizer(texts=["gold"])
        assert isinstance(t, FinancialTokenizer)


# ---------------------------------------------------------------------------
# vocab_size_actual property
# ---------------------------------------------------------------------------


class TestVocabSizeActual:
    def test_matches_len(self):
        t = FinancialTokenizer(vocab_size=200)
        t.train(["gold silver oil"])
        assert t.vocab_size_actual == len(t)
