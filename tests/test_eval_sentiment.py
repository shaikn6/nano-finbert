"""Tests for the committed sentiment eval set and scripts/eval_sentiment.py."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from finbert.signals import SignalExtractor

ROOT = Path(__file__).parent.parent
EVAL_PATH = ROOT / "data" / "eval" / "headlines_sentiment.json"
SAMPLE_PATH = ROOT / "data" / "samples" / "financial_phrases.json"

_spec = importlib.util.spec_from_file_location("eval_sentiment", ROOT / "scripts/eval_sentiment.py")
eval_sentiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_sentiment)


def test_eval_set_is_balanced_unique_and_disjoint_from_sample_set():
    examples = eval_sentiment.load_examples(EVAL_PATH)
    texts = [ex["text"] for ex in examples]
    assert Counter(ex["sentiment"] for ex in examples) == {
        "positive": 30,
        "negative": 30,
        "neutral": 30,
    }
    assert len(set(texts)) == len(texts)
    assert not set(texts) & {ex["text"] for ex in json.loads(SAMPLE_PATH.read_text())}


def test_load_examples_rejects_unknown_label(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"text": "x", "sentiment": "bullish"}]))
    with pytest.raises(ValueError, match="unknown sentiment label"):
        eval_sentiment.load_examples(bad)


def test_evaluate_scores_perfect_and_inverted_predictions():
    examples = [
        {"text": "a", "sentiment": "positive"},
        {"text": "b", "sentiment": "negative"},
        {"text": "c", "sentiment": "neutral"},
    ]

    def stub(mapping):
        return SimpleNamespace(
            extract_batch=lambda texts: [SimpleNamespace(sentiment=mapping[t]) for t in texts]
        )

    right = eval_sentiment.evaluate(
        stub({"a": "positive", "b": "negative", "c": "neutral"}), examples
    )
    wrong = eval_sentiment.evaluate(
        stub({"a": "negative", "b": "neutral", "c": "positive"}), examples
    )
    assert right["accuracy"] == 1.0 and right["macro_f1"] == 1.0
    assert wrong["accuracy"] == 0.0 and wrong["macro_f1"] == 0.0


def test_shipped_checkpoint_beats_chance_on_eval_set():
    report = eval_sentiment.evaluate(
        SignalExtractor.pretrained(), eval_sentiment.load_examples(EVAL_PATH)
    )
    # Chance on a balanced 3-class set is 0.333.
    assert report["accuracy"] > 0.5
    assert report["macro_f1"] > 0.5


def test_cli_refuses_model_without_tokenizer(monkeypatch):
    monkeypatch.setattr("sys.argv", ["eval_sentiment.py", "--model", "some.pt"])
    with pytest.raises(SystemExit) as exc:
        eval_sentiment.main()
    assert exc.value.code == 2
