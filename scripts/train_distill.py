"""
Train the NanoFinBERT architecture on original training sentences plus
teacher-labelled (pseudo-labelled) text, evaluated on the fixed PhraseBank split.

Differences from scripts/train.py:
  - the BPE tokenizer is built on TRAINING texts only (no val/test text);
  - val/test are the exact sentences of make_split_dataloaders(seed=42);
  - extra pseudo-labelled examples are added to the training set only.

Usage:
    PYTHONPATH=src python scripts/train_distill.py \
        --pseudo-path <work>/pseudo_labeled.json --out-dir <work>/run1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from distill_label import CLASSES, fixed_split  # noqa: E402

from finbert.dataset import SENTIMENT_LABELS, FinancialPhraseDataset, _collate, _wrap_loader  # noqa: E402
from finbert.metrics import classification_report, format_report  # noqa: E402
from finbert.model import NanoFinBERT  # noqa: E402
from finbert.tokenizer import FinancialTokenizer, build_default_tokenizer  # noqa: E402
from finbert.train import TrainConfig, Trainer  # noqa: E402


class TargetDS(Dataset):
    """Adds a 3-way target distribution: teacher probs if given, else one-hot of the hard label."""

    def __init__(self, base, probs=None):
        self.base, self.probs = base, probs

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = dict(self.base[i])
        if self.probs is not None:
            item["target"] = torch.tensor(self.probs[i], dtype=torch.float)
        else:
            item["target"] = torch.nn.functional.one_hot(item["label"], 3).float()
        return item


def _collate_t(batch):
    out = _collate(batch)
    out["target"] = torch.stack([b["target"] for b in batch])
    return out


class SoftTrainer(Trainer):
    """Training step against target distributions (soft CE = KL up to a constant, temperature 1).
    Hard-labelled rows carry one-hot targets, so they reduce to ordinary cross-entropy."""

    def _train_step(self, batch):
        d = self.config.device
        logits = self.model(batch["input_ids"].to(d), batch["attention_mask"].to(d))[
            "sentiment_logits"
        ]
        target = batch["target"].to(d)
        loss = -(target * torch.log_softmax(logits, -1)).sum(-1).mean()
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        ).item()
        self.optimiser.step()
        self.scheduler.step()
        self.global_step += 1
        return loss.item(), grad_norm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/financial_phrasebank.json")
    ap.add_argument(
        "--pseudo-path", default=None, help="pseudo_labeled.json; omit for no extra data"
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=64, help="padding/truncation length")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--soft",
        action="store_true",
        help="train pseudo rows against teacher probs (needs 'probs')",
    )
    ap.add_argument(
        "--tok-cache-dir",
        default=None,
        help="cache built tokenizers here, keyed by hash of training texts",
    )
    ap.add_argument("--tok-path", default=None, help="load this exact tokenizer file")
    ap.add_argument("--init-ckpt", default=None, help="pretrained encoder checkpoint")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples = json.load(open(args.data_path, encoding="utf-8"))
    train_idx, val_idx, test_idx = fixed_split(len(samples))
    pseudo = json.load(open(args.pseudo_path, encoding="utf-8")) if args.pseudo_path else []
    train_texts = [samples[i]["text"] for i in train_idx] + [p["text"] for p in pseudo]
    print(
        f"train={len(train_idx)} original + {len(pseudo)} pseudo | val={len(val_idx)} test={len(test_idx)}"
    )

    # Tokenizer on TRAINING texts only
    t0 = time.time()
    cache = None
    if args.tok_cache_dir:
        Path(args.tok_cache_dir).mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1("\n".join(train_texts).encode("utf-8")).hexdigest()[:16]
        cache = Path(args.tok_cache_dir) / f"tok_{key}.json"
    if args.tok_path:
        tokenizer = FinancialTokenizer.load(args.tok_path)
        print(f"tokenizer loaded from {args.tok_path}: {tokenizer}")
    elif cache is not None and cache.exists():
        tokenizer = FinancialTokenizer.load(str(cache))
        print(f"tokenizer loaded from cache {cache}: {tokenizer} ({time.time() - t0:.0f}s)")
    else:
        tokenizer = build_default_tokenizer(train_texts)
        if cache is not None:
            tokenizer.save(str(cache))
        print(f"tokenizer built: {tokenizer} ({time.time() - t0:.0f}s)")

    # Datasets: original file in file order (so split indices match), pseudo as a second dataset
    full = FinancialPhraseDataset(args.data_path, tokenizer, max_length=args.max_length)
    pseudo_ds = FinancialPhraseDataset.__new__(FinancialPhraseDataset)
    pseudo_ds.tokenizer, pseudo_ds.max_length = tokenizer, args.max_length
    pseudo_ds.examples = [
        {
            "text": p["text"],
            "sentiment": p["label"],
            "label": SENTIMENT_LABELS[p["label"]],
            "entities": [],
        }
        for p in pseudo
    ]
    parts = [TargetDS(Subset(full, train_idx))]
    if pseudo:
        parts.append(TargetDS(pseudo_ds, [p["probs"] for p in pseudo] if args.soft else None))
    train_set = ConcatDataset(parts)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_t
    )
    val_loader = _wrap_loader(Subset(full, val_idx), args.batch_size, shuffle=False)
    test_loader = _wrap_loader(Subset(full, test_idx), args.batch_size, shuffle=False)

    torch.manual_seed(args.seed)
    model = NanoFinBERT(vocab_size=8000, hidden_dim=128, num_layers=4, num_heads=4, max_seq_len=256)
    print(f"Parameters: {model.count_parameters():,}")
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=True)
        res = model.load_state_dict(ck["state_dict"], strict=False)
        assert not res.unexpected_keys, res.unexpected_keys
        n_loaded = len(ck["state_dict"])
        fresh = res.missing_keys
        print(
            f"PRETRAINED: loaded {n_loaded} tensors of {len(model.state_dict())}; fresh init: {len(fresh)}"
        )
        print("fresh tensors:", fresh)
        w = model.state_dict()
        assert all(torch.equal(w[k], v) for k, v in ck["state_dict"].items())

    config = TrainConfig(
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        num_epochs=args.epochs,
        checkpoint_dir=str(out / "ckpt"),
        save_every_n_steps=0,
        eval_every_n_epochs=1,
        log_every_n_steps=50,
        device=args.device,
    )
    trainer = SoftTrainer(
        model, tokenizer, train_loader, val_loader, config, test_loader=test_loader
    )
    t1 = time.time()
    history = trainer.train()
    train_secs = time.time() - t1

    # Best checkpoint by validation accuracy, then ONE test evaluation
    best = Path(config.checkpoint_dir) / "best_model.pt"
    best_epoch = torch.load(best, map_location="cpu", weights_only=True)["epoch"]
    trainer.load_checkpoint(str(best))
    val_m = trainer.evaluate_split(val_loader)
    test_m = trainer.evaluate_split(test_loader)
    print(f"\nbest val acc {val_m['accuracy']:.4f} at epoch {best_epoch}")
    print("STUDENT test results (fixed split, best-val checkpoint)")
    print(format_report(test_m["report"]))

    # Reference: TF-IDF + logistic regression on the 2,295 original training sentences
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    xtr = vec.fit_transform([samples[i]["text"] for i in train_idx])
    ytr = [SENTIMENT_LABELS[samples[i]["sentiment"]] for i in train_idx]
    clf = LogisticRegression(max_iter=2000).fit(xtr, ytr)
    tp = torch.tensor(clf.predict(vec.transform([samples[i]["text"] for i in test_idx])))
    ty = torch.tensor([SENTIMENT_LABELS[samples[i]["sentiment"]] for i in test_idx])
    tfidf_rep = classification_report(tp, ty, 3, CLASSES)
    print("\nTF-IDF + LogReg test results")
    print(format_report(tfidf_rep))

    json.dump(
        {
            "args": vars(args),
            "params": model.count_parameters(),
            "train_seconds": train_secs,
            "best_epoch": best_epoch,
            "best_val_acc": val_m["accuracy"],
            "history": history,
            "student_test": test_m["report"],
            "tfidf_test": tfidf_rep,
        },
        open(out / "results.json", "w"),
        indent=2,
        default=float,
    )


if __name__ == "__main__":
    main()
