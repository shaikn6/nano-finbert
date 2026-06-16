"""
Tests for FinancialTokenizer.

Covers:
  - Training on financial text
  - Encode / decode roundtrip
  - Special token handling
  - Padding and truncation
  - Financial vocabulary coverage
  - Save / load persistence
"""

from __future__ import annotations

from pathlib import Path

from finbert.tokenizer import (
    FinancialTokenizer,
    build_default_tokenizer,
    vocab_stats,
)


class TestSpecialTokens:
    def test_special_token_ids_are_fixed(self):
        t = FinancialTokenizer()
        assert t.SPECIAL_TOKENS["[PAD]"] == 0
        assert t.SPECIAL_TOKENS["[UNK]"] == 1
        assert t.SPECIAL_TOKENS["[CLS]"] == 2
        assert t.SPECIAL_TOKENS["[SEP]"] == 3
        assert t.SPECIAL_TOKENS["[MASK]"] == 4

    def test_special_tokens_present_before_training(self):
        t = FinancialTokenizer()
        for token, idx in t.SPECIAL_TOKENS.items():
            assert token in t.token_to_id
            assert t.token_to_id[token] == idx

    def test_id_to_token_reverse_mapping(self):
        t = FinancialTokenizer()
        for token, idx in t.SPECIAL_TOKENS.items():
            assert t.id_to_token[idx] == token


class TestTraining:
    def test_train_increases_vocab(self):
        t = FinancialTokenizer(vocab_size=200)
        initial_size = len(t.token_to_id)
        texts = [
            "gold prices rose after Fed rate decision",
            "bitcoin fell below key support level",
            "earnings beat expectations for the quarter",
        ]
        t.train(texts)
        assert len(t.token_to_id) > initial_size

    def test_train_sets_trained_flag(self):
        t = FinancialTokenizer(vocab_size=200)
        assert not t._trained
        t.train(["gold silver copper"])
        assert t._trained

    def test_vocab_does_not_exceed_vocab_size(self, trained_tokenizer):
        assert len(trained_tokenizer.token_to_id) <= trained_tokenizer.vocab_size

    def test_build_default_tokenizer_returns_trained(self):
        t = build_default_tokenizer()
        assert t._trained
        assert len(t) > len(t.SPECIAL_TOKENS)


class TestEncode:
    def test_encode_returns_required_keys(self, trained_tokenizer):
        result = trained_tokenizer.encode("gold prices rose")
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_encode_starts_with_cls(self, trained_tokenizer):
        result = trained_tokenizer.encode("gold rose")
        assert result["input_ids"][0] == FinancialTokenizer.SPECIAL_TOKENS["[CLS]"]

    def test_encode_ends_with_sep_before_padding(self, trained_tokenizer):
        result = trained_tokenizer.encode("gold rose", max_length=32)
        ids = result["input_ids"]
        # Find last non-pad token — should be SEP
        sep_id = FinancialTokenizer.SPECIAL_TOKENS["[SEP]"]
        pad_id = FinancialTokenizer.SPECIAL_TOKENS["[PAD]"]
        non_pad = [i for i in ids if i != pad_id]
        assert non_pad[-1] == sep_id

    def test_encode_length_equals_max_length_when_padded(self, trained_tokenizer):
        result = trained_tokenizer.encode("short text", max_length=64, padding=True)
        assert len(result["input_ids"]) == 64
        assert len(result["attention_mask"]) == 64

    def test_encode_truncation(self, trained_tokenizer):
        long_text = " ".join(["gold"] * 300)
        result = trained_tokenizer.encode(long_text, max_length=32, truncation=True, padding=True)
        assert len(result["input_ids"]) == 32

    def test_attention_mask_ones_for_real_tokens(self, trained_tokenizer):
        result = trained_tokenizer.encode("gold silver", max_length=32, padding=True)
        ids = result["input_ids"]
        mask = result["attention_mask"]
        pad_id = FinancialTokenizer.SPECIAL_TOKENS["[PAD]"]
        for i, (id_, m) in enumerate(zip(ids, mask)):
            if id_ == pad_id:
                assert m == 0, f"Padding token at {i} should have mask=0"
            else:
                assert m == 1, f"Real token at {i} should have mask=1"

    def test_encode_no_padding(self, trained_tokenizer):
        result = trained_tokenizer.encode("gold", max_length=32, padding=False)
        pad_id = FinancialTokenizer.SPECIAL_TOKENS["[PAD]"]
        assert pad_id not in result["input_ids"]

    def test_encode_empty_text(self, trained_tokenizer):
        # Should not raise; returns [CLS][SEP] + padding
        result = trained_tokenizer.encode("", max_length=16, padding=True)
        assert len(result["input_ids"]) == 16
        assert result["input_ids"][0] == FinancialTokenizer.SPECIAL_TOKENS["[CLS]"]

    def test_encode_financial_ticker(self, trained_tokenizer):
        # Ticker symbols should be handled without error
        result = trained_tokenizer.encode("$AAPL rose 3% today")
        assert len(result["input_ids"]) > 0

    def test_ids_are_within_vocab_range(self, trained_tokenizer):
        result = trained_tokenizer.encode("bitcoin ethereum crypto defi")
        for id_ in result["input_ids"]:
            assert 0 <= id_ < trained_tokenizer.vocab_size


class TestDecode:
    def test_decode_skips_special_tokens_by_default(self, trained_tokenizer):
        result = trained_tokenizer.encode("gold rose", max_length=32, padding=True)
        decoded = trained_tokenizer.decode(result["input_ids"])
        assert "[CLS]" not in decoded
        assert "[SEP]" not in decoded
        assert "[PAD]" not in decoded

    def test_decode_includes_special_tokens_when_requested(self, trained_tokenizer):
        result = trained_tokenizer.encode("gold", max_length=16, padding=False)
        decoded = trained_tokenizer.decode(result["input_ids"], skip_special_tokens=False)
        assert "[CLS]" in decoded
        assert "[SEP]" in decoded

    def test_decode_returns_string(self, trained_tokenizer):
        result = trained_tokenizer.encode("oil prices fell")
        decoded = trained_tokenizer.decode(result["input_ids"])
        assert isinstance(decoded, str)

    def test_decode_unknown_id_returns_unk(self, trained_tokenizer):
        decoded = trained_tokenizer.decode([99999], skip_special_tokens=False)
        assert "[UNK]" in decoded


class TestPersistence:
    def test_save_and_load_roundtrip(self, trained_tokenizer, tmp_path):
        path = str(tmp_path / "tokenizer.json")
        trained_tokenizer.save(path)

        loaded = FinancialTokenizer.load(path)
        assert loaded.vocab_size == trained_tokenizer.vocab_size
        assert loaded.token_to_id == trained_tokenizer.token_to_id
        assert loaded._trained == trained_tokenizer._trained

    def test_loaded_tokenizer_encodes_same(self, trained_tokenizer, tmp_path):
        path = str(tmp_path / "tokenizer.json")
        trained_tokenizer.save(path)
        loaded = FinancialTokenizer.load(path)

        text = "gold futures fell on dollar strength"
        orig = trained_tokenizer.encode(text, max_length=32)
        reloaded = loaded.encode(text, max_length=32)
        assert orig["input_ids"] == reloaded["input_ids"]

    def test_save_creates_parent_dirs(self, trained_tokenizer, tmp_path):
        path = str(tmp_path / "nested" / "deep" / "tokenizer.json")
        trained_tokenizer.save(path)
        assert Path(path).exists()


class TestVocabStats:
    def test_vocab_stats_returns_dict(self, trained_tokenizer):
        stats = vocab_stats(trained_tokenizer)
        assert isinstance(stats, dict)
        assert "total_tokens" in stats
        assert "special_tokens" in stats

    def test_special_token_count_is_five(self, trained_tokenizer):
        stats = vocab_stats(trained_tokenizer)
        assert stats["special_tokens"] == 5


class TestRepr:
    def test_repr_contains_vocab_size(self, trained_tokenizer):
        r = repr(trained_tokenizer)
        assert "FinancialTokenizer" in r
        assert "trained" in r
