"""
Shared pytest fixtures for nano-finbert tests.

All fixtures use CPU-only, random-weight models so tests run
without a GPU and without a trained checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from finbert.model import NanoFinBERT
from finbert.tokenizer import FinancialTokenizer, build_default_tokenizer

# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_model() -> NanoFinBERT:
    """A minimal NanoFinBERT (2 layers, 64 dim) with random weights — fast for tests."""
    model = NanoFinBERT(
        vocab_size=500,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=64,
        dropout=0.0,
    )
    model.eval()
    return model


@pytest.fixture(scope="session")
def default_model() -> NanoFinBERT:
    """Full-size NanoFinBERT with random weights."""
    model = NanoFinBERT()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tokenizer fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def trained_tokenizer() -> FinancialTokenizer:
    """A tokenizer trained on a small financial corpus."""
    texts = [
        "Apple reported record quarterly earnings beating analyst estimates",
        "Bitcoin dropped below 60000 as regulatory pressure mounted",
        "Gold futures fell 2.3% amid dollar strength and Fed rate concerns",
        "SpaceX IPO filing sent aerospace stocks soaring to record highs",
        "Tesla stock rallied after strong delivery numbers exceeded guidance",
        "Federal Reserve raised interest rates by 25 basis points",
        "Oil prices surged following OPEC supply cut announcement",
        "Nvidia semiconductor revenue exceeded expectations by 30 percent",
        "Microsoft Azure cloud revenue grew 29% year over year",
        "Copper prices declined on weak Chinese manufacturing data",
        "Ethereum staking yield attracted institutional investors",
        "Goldman Sachs upgraded Apple to buy with 200 dollar price target",
        "Wheat futures jumped on drought concerns in major producing regions",
        "ECB signaled potential rate cuts amid eurozone growth slowdown",
        "Amazon Web Services posted record operating income this quarter",
    ]
    return build_default_tokenizer(texts)


@pytest.fixture(scope="session")
def tiny_tokenizer() -> FinancialTokenizer:
    """Minimal tokenizer with vocab_size=500 for fast tests."""
    texts = ["gold silver copper oil bitcoin ethereum dollar euro"]
    tokenizer = FinancialTokenizer(vocab_size=500)
    tokenizer.train(texts)
    return tokenizer


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_data_path(tmp_path_factory) -> Path:
    """Write a small in-memory dataset to a temp file and return the path."""
    samples = [
        {
            "text": "SpaceX IPO filing sent aerospace stocks soaring",
            "sentiment": "positive",
            "entities": ["SpaceX", "aerospace"],
        },
        {
            "text": "Bitcoin dropped below $60,000 as regulatory pressure mounted",
            "sentiment": "negative",
            "entities": ["Bitcoin", "$60,000"],
        },
        {
            "text": "Gold futures fell 2.3% amid dollar strength",
            "sentiment": "negative",
            "entities": ["Gold", "dollar"],
        },
        {
            "text": "Apple reported record quarterly revenue of $119.6 billion",
            "sentiment": "positive",
            "entities": ["Apple", "revenue"],
        },
        {
            "text": "Federal Reserve held interest rates steady at current levels",
            "sentiment": "neutral",
            "entities": ["Federal Reserve"],
        },
        {
            "text": "Tesla stock fell 8% after disappointing delivery numbers",
            "sentiment": "negative",
            "entities": ["Tesla"],
        },
        {
            "text": "Nvidia reported another quarter of record revenue growth",
            "sentiment": "positive",
            "entities": ["Nvidia"],
        },
        {
            "text": "Oil prices remained flat amid mixed demand signals",
            "sentiment": "neutral",
            "entities": ["Oil"],
        },
    ]
    tmp = tmp_path_factory.mktemp("data")
    path = tmp / "financial_phrases.json"
    path.write_text(json.dumps(samples), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Input tensor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_batch_small() -> dict:
    """A tiny (batch=2, seq=16) tensor batch for fast forward-pass tests."""
    return {
        "input_ids": torch.randint(0, 500, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
    }


@pytest.fixture
def sample_batch() -> dict:
    """A standard (batch=4, seq=64) tensor batch."""
    return {
        "input_ids": torch.randint(0, 8000, (4, 64)),
        "attention_mask": torch.ones(4, 64, dtype=torch.long),
    }
