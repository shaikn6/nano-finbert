"""
Train NanoFinBERT from scratch on the MIT-licensed Twitter Financial News
Sentiment dataset and write the checkpoint that ships with the repo.

Dataset: zeroshot/twitter-financial-news-sentiment (MIT, 11,932 finance tweets,
labels bearish / bullish / neutral). The dataset is downloaded at a pinned
revision and is not redistributed; only the trained weights are committed.

Split (all seeded, test scored exactly once, at the end):
  - test  = the dataset's official validation file, minus any text also in train
  - train / val = a stratified 90 / 10 split of the dataset's official train file

The BPE tokenizer is built on training text only. The checkpoint is chosen by
validation macro-F1 (classes are imbalanced, so accuracy alone would favour a model
that never predicts "negative"). A TF-IDF + logistic regression baseline is fitted on the
same train split (C chosen on val) and scored on the same test split.

Usage:
    PYTHONPATH=src python scripts/train_tfns.py
    PYTHONPATH=src python scripts/train_tfns.py --no-test      # dev run, never touches test
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from pins import revision  # noqa: E402

from finbert.dataset import SENTIMENT_LABELS, FinancialPhraseDataset, _wrap_loader  # noqa: E402
from finbert.metrics import classification_report, format_report  # noqa: E402
from finbert.model import NanoFinBERT  # noqa: E402
from finbert.tokenizer import build_default_tokenizer  # noqa: E402
from finbert.train import TrainConfig, Trainer  # noqa: E402

REPO_ID = "zeroshot/twitter-financial-news-sentiment"
# Dataset labels: 0 Bearish, 1 Bullish, 2 Neutral
LABEL_MAP = {0: "negative", 1: "positive", 2: "neutral"}
CLASSES = ["negative", "neutral", "positive"]  # index order of SENTIMENT_LABELS
_URL = re.compile(r"https?://\S+")
WEIGHTS_DIR = Path(__file__).parent.parent / "src" / "finbert" / "weights"


def clean(text: str) -> str:
    """Drop t.co links (random strings carrying no sentiment) and collapse whitespace."""
    return " ".join(_URL.sub(" ", text).split())


def read_split(name: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO_ID, name, repo_type="dataset", revision=revision(REPO_ID))
    with open(path, newline="", encoding="utf-8") as f:
        rows = [(clean(r["text"]), LABEL_MAP[int(r["label"])]) for r in csv.DictReader(f)]
    return [{"text": t, "sentiment": s} for t, s in rows if t]


def dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = r["text"].lower()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def stratified_split(rows: list[dict], val_frac: float, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    train, val = [], []
    for cls in CLASSES:
        group = [r for r in rows if r["sentiment"] == cls]
        rng.shuffle(group)
        n_val = round(len(group) * val_frac)
        val += group[:n_val]
        train += group[n_val:]
    rng.shuffle(train)
    return train, val


def tfidf_baseline(train, val, test) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    y = lambda rows: [SENTIMENT_LABELS[r["sentiment"]] for r in rows]  # noqa: E731
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_train = vec.fit_transform([r["text"] for r in train])
    x_val = vec.transform([r["text"] for r in val])
    best = None
    for c in (0.3, 1.0, 3.0, 10.0, 30.0):
        clf = LogisticRegression(C=c, max_iter=2000).fit(x_train, y(train))
        acc = float((clf.predict(x_val) == y(val)).mean())
        if best is None or acc > best[0]:
            best = (acc, c, clf)
    val_acc, c, clf = best
    out = {"C": c, "val_accuracy": val_acc}
    if test is not None:
        pred = clf.predict(vec.transform([r["text"] for r in test]))
        rep = classification_report(torch.tensor(pred), torch.tensor(y(test)), 3, CLASSES)
        out.update(accuracy=rep["accuracy"], macro_f1=rep["macro_f1"])
    return out


def majority_baseline(train, test) -> dict:
    top = Counter(r["sentiment"] for r in train).most_common(1)[0][0]
    preds = torch.full((len(test),), SENTIMENT_LABELS[top])
    labels = torch.tensor([SENTIMENT_LABELS[r["sentiment"]] for r in test])
    rep = classification_report(preds, labels, 3, CLASSES)
    return {"class": top, "accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--class-weights", action="store_true", help="inverse-frequency loss weights")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(WEIGHTS_DIR))
    ap.add_argument("--no-test", action="store_true", help="dev run: never score the test split")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    train_all = dedupe(read_split("sent_train.csv"))
    test = dedupe(read_split("sent_valid.csv"))
    train_keys = {r["text"].lower() for r in train_all}
    n_test_raw = len(test)
    test = [r for r in test if r["text"].lower() not in train_keys]
    train, val = stratified_split(train_all, 0.10, args.seed)
    print(
        f"train {len(train)} | val {len(val)} | test {len(test)} (dropped {n_test_raw - len(test)} overlapping/duplicate)"
    )
    for name, rows in (("train", train), ("val", val), ("test", test)):
        print(f"  {name}: {dict(Counter(r['sentiment'] for r in rows))}")

    tokenizer = build_default_tokenizer([r["text"] for r in train])
    print(f"tokenizer: {tokenizer}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        loaders = {}
        for name, rows in (("train", train), ("val", val), ("test", test)):
            p = Path(tmp) / f"{name}.json"
            p.write_text(json.dumps(rows), encoding="utf-8")
            ds = FinancialPhraseDataset(p, tokenizer, max_length=args.max_len)
            loaders[name] = _wrap_loader(ds, args.batch_size, shuffle=(name == "train"))

        model = NanoFinBERT(dropout=args.dropout)  # 128 hidden, 4 layers, 4 heads, 8000 vocab
        config = TrainConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_epochs=args.epochs,
            checkpoint_dir=str(Path(tmp) / "ckpt"),
            save_every_n_steps=0,
            log_every_n_steps=100,
            device="cpu",
        )
        trainer = Trainer(model, tokenizer, loaders["train"], loaders["val"], config)
        if args.class_weights:
            counts = Counter(SENTIMENT_LABELS[r["sentiment"]] for r in train)
            w = torch.tensor([len(train) / (3 * counts[i]) for i in range(3)], dtype=torch.float)
            trainer.criterion = torch.nn.CrossEntropyLoss(weight=w)
        t0 = time.time()
        history, best_f1, best_epoch, best_state = [], -1.0, 0, None
        for epoch in range(1, args.epochs + 1):
            train_loss = trainer._train_epoch(epoch)
            vm = trainer.evaluate_split(loaders["val"])
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": vm["loss"],
                    "val_accuracy": vm["accuracy"],
                    "val_macro_f1": vm["macro_f1"],
                }
            )
            print(
                f"epoch {epoch:2d} | train_loss={train_loss:.4f} | val_loss={vm['loss']:.4f} | "
                f"val_acc={vm['accuracy']:.3f} | val_macro_f1={vm['macro_f1']:.3f}",
                flush=True,
            )
            if vm["macro_f1"] > best_f1:
                best_f1, best_epoch = vm["macro_f1"], epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        train_seconds = time.time() - t0

        # Keep the epoch with the best validation macro-F1; the optimiser state is not shipped.
        model.load_state_dict(best_state)
        val_metrics = trainer.evaluate_split(loaders["val"])

        result = {
            "dataset": REPO_ID,
            "dataset_revision": revision(REPO_ID),
            "dataset_license": "MIT",
            "label_map": {"bearish": "negative", "bullish": "positive", "neutral": "neutral"},
            "seed": args.seed,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "class_weights": args.class_weights,
            "max_len": args.max_len,
            "train_seconds": round(train_seconds, 1),
            "split": {"train": len(train), "val": len(val), "test": len(test)},
            "val": {"accuracy": val_metrics["accuracy"], "macro_f1": val_metrics["macro_f1"]},
            "history": history,
        }
        print("\nValidation (best checkpoint):")
        print(format_report(val_metrics["report"]))

        if args.no_test:
            result["baseline_tfidf_logreg"] = tfidf_baseline(train, val, None)
            print(f"\n[dev run: test split not scored] {result['baseline_tfidf_logreg']}")
            return

        # Test split is scored exactly once, after the checkpoint is fixed.
        test_metrics = trainer.evaluate_split(loaders["test"])
        result["test"] = {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "report": test_metrics["report"],
        }
        result["baseline_majority"] = majority_baseline(train, test)
        result["baseline_tfidf_logreg"] = tfidf_baseline(train, val, test)
        print("\nHeld-out test set (scored once):")
        print(format_report(test_metrics["report"]))
        print(f"majority baseline: {result['baseline_majority']}")
        print(f"tfidf+logreg baseline: {result['baseline_tfidf_logreg']}")

        torch.save(
            {
                "config": model.config,
                "model_state_dict": model.state_dict(),
                "metadata": {
                    k: result[k]
                    for k in ("dataset", "dataset_revision", "seed", "epochs", "best_epoch")
                },
            },
            out_dir / "nano_finbert_tfns.pt",
        )
        tokenizer.save(str(out_dir / "tokenizer.json"))
        (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote {out_dir}/nano_finbert_tfns.pt, tokenizer.json, metrics.json")


if __name__ == "__main__":
    main()
