"""
Score the sentiment head on a labeled JSON file and print accuracy / macro-F1.

Sentiment is the only learned output of nano-finbert; entities, sectors, event type and
impact are rule-based, so this is the one capability that can be measured against labels.

The default file, data/eval/headlines_sentiment.json, holds 90 hand-written headlines
(30 per class) that are in neither the training data nor data/samples/. Any file with
the same format ([{"text": ..., "sentiment": positive|negative|neutral}, ...]) works.

Usage:
    PYTHONPATH=src python scripts/eval_sentiment.py
    PYTHONPATH=src python scripts/eval_sentiment.py --data data/samples/financial_phrases.json
    PYTHONPATH=src python scripts/eval_sentiment.py --model checkpoints/best_model.pt \
        --tokenizer checkpoints/tokenizer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finbert.dataset import SENTIMENT_LABELS
from finbert.metrics import classification_report, format_report
from finbert.signals import DEFAULT_MODEL_PATH, DEFAULT_TOKENIZER_PATH, SignalExtractor

DEFAULT_DATA = Path(__file__).parent.parent / "data" / "eval" / "headlines_sentiment.json"


def load_examples(path: str | Path) -> list[dict]:
    """Read and validate a labeled file; fail loudly on a bad label instead of scoring it."""
    examples = json.loads(Path(path).read_text())
    for i, ex in enumerate(examples):
        if ex.get("sentiment") not in SENTIMENT_LABELS:
            raise ValueError(f"{path}[{i}]: unknown sentiment label {ex.get('sentiment')!r}")
    return examples


def evaluate(extractor: SignalExtractor, examples: list[dict]) -> dict:
    """Run the extractor over the examples and return classification_report()."""
    signals = extractor.extract_batch([ex["text"] for ex in examples])
    predictions = torch.tensor([SENTIMENT_LABELS[s.sentiment] for s in signals])
    labels = torch.tensor([SENTIMENT_LABELS[ex["sentiment"]] for ex in examples])
    names = sorted(SENTIMENT_LABELS, key=SENTIMENT_LABELS.get)
    return classification_report(predictions, labels, len(names), names)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the sentiment head on a labeled file.")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="Labeled JSON file")
    p.add_argument("--model", default=None, help="Checkpoint (default: the shipped one)")
    p.add_argument("--tokenizer", default=None, help="Tokenizer matching --model")
    p.add_argument(
        "--json", action="store_true", dest="json_output", help="Print the report as JSON"
    )
    args = p.parse_args()

    # A checkpoint is only meaningful with the tokenizer it was trained with.
    model, tokenizer = args.model, args.tokenizer
    if model and not tokenizer:
        p.error("--model needs --tokenizer: a checkpoint scored with the wrong vocabulary is noise")
    extractor = (
        SignalExtractor(model_path=model, tokenizer_path=tokenizer)
        if model
        else SignalExtractor(
            model_path=str(DEFAULT_MODEL_PATH), tokenizer_path=str(DEFAULT_TOKENIZER_PATH)
        )
    )

    examples = load_examples(args.data)
    report = evaluate(extractor, examples)
    if args.json_output:
        print(json.dumps(report, indent=2))
    else:
        print(f"{len(examples)} examples from {args.data}\n")
        print(format_report(report))


if __name__ == "__main__":
    main()
