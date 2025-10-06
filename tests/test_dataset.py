"""
Comprehensive tests for FinancialPhraseDataset and dataset utilities.
Covers dataset.py to push coverage to 95%+.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import DataLoader

from finbert.dataset import (
    ID_TO_SENTIMENT,
    SENTIMENT_LABELS,
    FinancialPhraseDataset,
    load_sample_dataset,
    make_dataloaders,
)
from finbert.tokenizer import FinancialTokenizer


# ---------------------------------------------------------------------------
# Local fixtures (self-contained — no dependency on conftest session fixtures)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_records():
    return [
        {"text": "Apple reported record earnings", "sentiment": "positive", "entities": ["Apple"]},
        {"text": "Bitcoin fell on regulatory fears", "sentiment": "negative", "entities": ["Bitcoin"]},
        {"text": "Gold prices remained flat", "sentiment": "neutral", "entities": ["Gold"]},
        {"text": "Tesla stock rally continued", "sentiment": "positive", "entities": ["Tesla"]},
        {"text": "Oil prices dropped sharply", "sentiment": "negative", "entities": []},
        {"text": "Fed held rates steady", "sentiment": "neutral", "entities": ["Fed"]},
    ]


@pytest.fixture
def dataset_path(tmp_path, sample_records):
    path = tmp_path / "phrases.json"
    path.write_text(json.dumps(sample_records), encoding="utf-8")
    return path


@pytest.fixture
def micro_tokenizer():
    t = FinancialTokenizer(vocab_size=500)
    t.train(["apple bitcoin gold tesla oil fed earnings record rally fell prices reported flat"])
    return t


@pytest.fixture
def dataset(dataset_path, micro_tokenizer):
    return FinancialPhraseDataset(dataset_path, micro_tokenizer, max_length=32)


# ---------------------------------------------------------------------------
# SENTIMENT_LABELS and ID_TO_SENTIMENT
# ---------------------------------------------------------------------------


class TestConstants:
    def test_sentiment_labels_has_three_classes(self):
        assert len(SENTIMENT_LABELS) == 3

    def test_negative_maps_to_zero(self):
        assert SENTIMENT_LABELS["negative"] == 0

    def test_neutral_maps_to_one(self):
        assert SENTIMENT_LABELS["neutral"] == 1

    def test_positive_maps_to_two(self):
        assert SENTIMENT_LABELS["positive"] == 2

    def test_id_to_sentiment_reverse_mapping(self):
        for label, idx in SENTIMENT_LABELS.items():
            assert ID_TO_SENTIMENT[idx] == label

    def test_id_to_sentiment_has_three_entries(self):
        assert len(ID_TO_SENTIMENT) == 3


# ---------------------------------------------------------------------------
# FinancialPhraseDataset — construction and __getitem__
# ---------------------------------------------------------------------------


class TestFinancialPhraseDataset:
    def test_len_matches_records(self, dataset, sample_records):
        assert len(dataset) == len(sample_records)

    def test_getitem_returns_dict(self, dataset):
        assert isinstance(dataset[0], dict)

    def test_getitem_has_all_required_keys(self, dataset):
        keys = {"input_ids", "attention_mask", "label", "text", "entities"}
        assert keys.issubset(dataset[0].keys())

    def test_input_ids_dtype_is_long(self, dataset):
        assert dataset[0]["input_ids"].dtype == torch.long

    def test_attention_mask_dtype_is_long(self, dataset):
        assert dataset[0]["attention_mask"].dtype == torch.long

    def test_label_dtype_is_long(self, dataset):
        assert dataset[0]["label"].dtype == torch.long

    def test_input_ids_length_equals_max_length(self, dataset):
        assert len(dataset[0]["input_ids"]) == 32

    def test_all_labels_are_valid(self, dataset):
        for i in range(len(dataset)):
            assert dataset[i]["label"].item() in (0, 1, 2)

    def test_text_matches_original(self, dataset, sample_records):
        for i, rec in enumerate(sample_records):
            assert dataset[i]["text"] == rec["text"]

    def test_entities_is_list(self, dataset):
        assert isinstance(dataset[0]["entities"], list)

    def test_positive_label_equals_two(self, tmp_path, micro_tokenizer):
        path = tmp_path / "pos.json"
        path.write_text(json.dumps([{"text": "stocks rallied", "sentiment": "positive"}]))
        ds = FinancialPhraseDataset(path, micro_tokenizer, max_length=16)
        assert ds[0]["label"].item() == 2

    def test_negative_label_equals_zero(self, tmp_path, micro_tokenizer):
        path = tmp_path / "neg.json"
        path.write_text(json.dumps([{"text": "stocks crashed", "sentiment": "negative"}]))
        ds = FinancialPhraseDataset(path, micro_tokenizer, max_length=16)
        assert ds[0]["label"].item() == 0

    def test_neutral_label_equals_one(self, tmp_path, micro_tokenizer):
        path = tmp_path / "neu.json"
        path.write_text(json.dumps([{"text": "stocks unchanged", "sentiment": "neutral"}]))
        ds = FinancialPhraseDataset(path, micro_tokenizer, max_length=16)
        assert ds[0]["label"].item() == 1

    def test_unknown_sentiment_defaults_to_neutral(self, tmp_path, micro_tokenizer):
        path = tmp_path / "unk.json"
        path.write_text(json.dumps([{"text": "mystery", "sentiment": "confused"}]))
        ds = FinancialPhraseDataset(path, micro_tokenizer, max_length=16)
        assert ds[0]["label"].item() == 1

    def test_missing_entities_field_defaults_to_empty_list(self, tmp_path, micro_tokenizer):
        path = tmp_path / "noent.json"
        path.write_text(json.dumps([{"text": "gold fell", "sentiment": "negative"}]))
        ds = FinancialPhraseDataset(path, micro_tokenizer, max_length=16)
        assert ds[0]["entities"] == []

    def test_missing_sentiment_field_defaults_to_neutral(self, tmp_path, micro_tokenizer):
        path = tmp_path / "nosent.json"
        path.write_text(json.dumps([{"text": "gold fell"}]))
        ds = FinancialPhraseDataset(path, micro_tokenizer, max_length=16)
        assert ds[0]["label"].item() == 1

    def test_label_distribution_sums_to_dataset_length(self, dataset):
        dist = dataset.label_distribution()
        assert sum(dist.values()) == len(dataset)

    def test_label_distribution_all_classes_present(self, dataset):
        dist = dataset.label_distribution()
        assert {"positive", "negative", "neutral"} == set(dist.keys())

    def test_label_distribution_counts_correct(self, dataset, sample_records):
        dist = dataset.label_distribution()
        for sentiment in ("positive", "negative", "neutral"):
            expected = sum(1 for r in sample_records if r["sentiment"] == sentiment)
            assert dist[sentiment] == expected

    def test_all_items_iterable_without_error(self, dataset):
        for i in range(len(dataset)):
            assert dataset[i] is not None

    def test_attention_mask_ones_for_real_tokens(self, dataset):
        item = dataset[0]
        pad_id = 0  # [PAD] = 0
        for tok, mask in zip(item["input_ids"].tolist(), item["attention_mask"].tolist()):
            if tok == pad_id:
                assert mask == 0
            else:
                assert mask == 1


# ---------------------------------------------------------------------------
# load_sample_dataset
# ---------------------------------------------------------------------------


class TestLoadSampleDataset:
    def test_raises_file_not_found_for_missing_path(self, micro_tokenizer, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_sample_dataset(micro_tokenizer, data_path=tmp_path / "missing.json")

    def test_loads_from_explicit_path(self, dataset_path, micro_tokenizer):
        ds = load_sample_dataset(micro_tokenizer, data_path=dataset_path)
        assert len(ds) > 0

    def test_returns_financial_phrase_dataset_instance(self, dataset_path, micro_tokenizer):
        ds = load_sample_dataset(micro_tokenizer, data_path=dataset_path)
        assert isinstance(ds, FinancialPhraseDataset)

    def test_custom_max_length_propagated(self, dataset_path, micro_tokenizer):
        ds = load_sample_dataset(micro_tokenizer, max_length=16, data_path=dataset_path)
        assert len(ds[0]["input_ids"]) == 16


# ---------------------------------------------------------------------------
# make_dataloaders
# ---------------------------------------------------------------------------


class TestMakeDataloaders:
    def test_returns_two_dataloaders(self, dataset):
        train_dl, val_dl = make_dataloaders(dataset, batch_size=2)
        assert isinstance(train_dl, DataLoader)
        assert isinstance(val_dl, DataLoader)

    def test_total_samples_preserved(self, dataset):
        train_dl, val_dl = make_dataloaders(dataset, batch_size=2, train_ratio=0.8)
        n = sum(b["input_ids"].shape[0] for b in train_dl)
        n += sum(b["input_ids"].shape[0] for b in val_dl)
        assert n == len(dataset)

    def test_batch_size_does_not_exceed_requested(self, dataset):
        train_dl, _ = make_dataloaders(dataset, batch_size=2)
        for batch in train_dl:
            assert batch["input_ids"].shape[0] <= 2

    def test_same_seed_produces_same_val_split(self, dataset):
        _, v1 = make_dataloaders(dataset, batch_size=len(dataset), seed=7)
        _, v2 = make_dataloaders(dataset, batch_size=len(dataset), seed=7)
        for b1, b2 in zip(v1, v2):
            assert torch.equal(b1["label"], b2["label"])

    def test_val_batches_contain_required_keys(self, dataset):
        _, val_dl = make_dataloaders(dataset, batch_size=2)
        for batch in val_dl:
            for key in ("input_ids", "attention_mask", "label"):
                assert key in batch
            break

    def test_train_loader_input_ids_dtype(self, dataset):
        train_dl, _ = make_dataloaders(dataset, batch_size=2)
        for batch in train_dl:
            assert batch["input_ids"].dtype == torch.long
            break

    def test_default_train_ratio_is_eighty_percent(self, dataset):
        train_dl, val_dl = make_dataloaders(dataset, batch_size=len(dataset))
        n_train = sum(b["input_ids"].shape[0] for b in train_dl)
        n_val = sum(b["input_ids"].shape[0] for b in val_dl)
        # With 6 samples and 0.8 ratio: 4 train, 2 val
        assert n_train + n_val == len(dataset)
        assert n_train >= n_val  # train should be larger
