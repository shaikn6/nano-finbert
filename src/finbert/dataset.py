"""
Financial dataset loader for NanoFinBERT.

Loads structured financial text examples with:
  - sentiment labels (positive / negative / neutral)
  - entity annotations (companies, tickers, commodities mentioned)

The built-in sample dataset (~250 examples) covers:
  IPOs, earnings, commodities, currencies, crypto, M&A, macro events.

Format of each record in financial_phrases.json:
    {
        "text": "Apple reported record quarterly revenue of $119.6 billion",
        "sentiment": "positive",
        "entities": ["Apple", "$AAPL", "revenue"]
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from finbert.tokenizer import FinancialTokenizer

# Sentiment label → integer index mapping
SENTIMENT_LABELS: dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

# Reverse mapping
ID_TO_SENTIMENT: dict[int, str] = {v: k for k, v in SENTIMENT_LABELS.items()}

# Default path to the built-in sample data
_DEFAULT_DATA_PATH = (
    Path(__file__).parent.parent.parent / "data" / "samples" / "financial_phrases.json"
)


class FinancialPhraseDataset(Dataset):
    """
    PyTorch Dataset wrapping financial text examples.

    Each item returned by __getitem__ is a dict:
        {
            "input_ids":      torch.LongTensor (seq_len,)
            "attention_mask": torch.LongTensor (seq_len,)
            "label":          torch.LongTensor scalar — 0/1/2 (neg/neu/pos)
            "text":           str — original text (for inspection)
            "entities":       list[str] — annotated entities
        }

    Args:
        data_path:   Path to JSON file containing list of examples.
        tokenizer:   Trained FinancialTokenizer instance.
        max_length:  Maximum token sequence length (default 256).
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: FinancialTokenizer,
        max_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, encoding="utf-8") as f:
            raw = json.load(f)

        self.examples: list[dict] = []
        for record in raw:
            sentiment = record.get("sentiment", "neutral").lower()
            if sentiment not in SENTIMENT_LABELS:
                sentiment = "neutral"
            self.examples.append(
                {
                    "text": record["text"],
                    "sentiment": sentiment,
                    "label": SENTIMENT_LABELS[sentiment],
                    "entities": record.get("entities", []),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        encoded = self.tokenizer.encode(
            example["text"],
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "label": torch.tensor(example["label"], dtype=torch.long),
            "text": example["text"],
            "entities": example["entities"],
        }

    def label_distribution(self) -> dict[str, int]:
        """Return the count of each sentiment class in the dataset."""
        counts: dict[str, int] = {"negative": 0, "neutral": 0, "positive": 0}
        for ex in self.examples:
            counts[ex["sentiment"]] += 1
        return counts


def load_sample_dataset(
    tokenizer: FinancialTokenizer,
    max_length: int = 256,
    data_path: str | Path | None = None,
) -> FinancialPhraseDataset:
    """
    Load the built-in sample financial phrases dataset.

    Args:
        tokenizer:  A FinancialTokenizer (trained or untrained — encoding still works).
        max_length: Maximum sequence length.
        data_path:  Override the default data path (optional).

    Returns:
        FinancialPhraseDataset ready for use with DataLoader.
    """
    path = Path(data_path) if data_path else _DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Sample data not found at {path}. "
            "Run from the project root or provide an explicit data_path."
        )
    return FinancialPhraseDataset(path, tokenizer, max_length)


def make_dataloaders(
    dataset: FinancialPhraseDataset,
    train_ratio: float = 0.8,
    batch_size: int = 16,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Split dataset into train/val and return DataLoaders.

    Args:
        dataset:     The full FinancialPhraseDataset.
        train_ratio: Fraction of data used for training (default 0.8).
        batch_size:  Batch size for both loaders.
        seed:        Random seed for reproducible splits.

    Returns:
        (train_loader, val_loader)
    """
    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader
