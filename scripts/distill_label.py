"""
Teacher labelling for the distillation experiment.

1. Evaluate the teacher on the fixed 491-sentence test split.
2. Stream unlabeled finance text, cut to 1-2 sentences, drop PhraseBank matches.
3. Label with the teacher, keep confident predictions, balance classes.

Usage:
    PYTHONPATH=src python scripts/distill_label.py --out-dir <work dir>
"""

from __future__ import annotations

from pins import revision
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import random_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finbert.metrics import classification_report, format_report  # noqa: E402

TEACHER = "9mark9/finbert-minilm-sentiment"
CLASSES = ["negative", "neutral", "positive"]
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def norm(t: str) -> str:
    t = re.sub(r"[^\w\s]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def fixed_split(n: int, seed: int = 42):
    """Index lists identical to make_split_dataloaders(train 0.7, val 0.15, seed 42)."""
    n_train = int(n * 0.7)
    n_test = int(n * max(0.0, 1.0 - 0.7 - 0.15))
    n_val = n - n_train - n_test
    parts = random_split(
        list(range(n)), [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed)
    )
    return [list(p.indices) for p in parts]


def cut_snippet(text: str, lo: int = 8, hi: int = 45) -> str | None:
    sents = [s.strip() for s in SENT_SPLIT.split(" ".join(text.split())) if s.strip()]
    if not sents:
        return None
    out = sents[0]
    if len(out.split()) < lo and len(sents) > 1:
        out = out + " " + sents[1]
    return out if lo <= len(out.split()) <= hi else None


@torch.no_grad()
def predict(tok, model, texts, device, bs=128):
    probs = []
    for i in range(0, len(texts), bs):
        enc = tok(
            texts[i : i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        probs.append(torch.softmax(model(**enc).logits.float(), -1).cpu())
    return torch.cat(probs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/financial_phrasebank.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-candidates", type=int, default=40000)
    ap.add_argument("--conf", type=float, default=0.90)
    ap.add_argument("--max-total", type=int, default=20000)
    ap.add_argument("--max-ratio", type=float, default=1.5)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    samples = json.load(open(args.data_path, encoding="utf-8"))
    pb_norm = {norm(s["text"]) for s in samples}
    _, _, test_idx = fixed_split(len(samples))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TEACHER, revision=revision(TEACHER))
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER, revision=revision(TEACHER)).to(device).eval()
    print("teacher id2label:", model.config.id2label, "device:", device)

    # --- 1. teacher on the fixed test split ---
    test_texts = [samples[i]["text"] for i in test_idx]
    test_y = torch.tensor([CLASSES.index(samples[i]["sentiment"]) for i in test_idx])
    pred = predict(tok, model, test_texts, device).argmax(-1)
    rep = classification_report(pred, test_y, 3, CLASSES)
    print("\nTEACHER on fixed test split (n=%d)" % len(test_idx))
    print(format_report(rep))
    json.dump(rep, open(out / "teacher_test_report.json", "w"), indent=2)

    # --- 2. stream + cut + leak guard ---
    t0 = time.time()
    ds = load_dataset("NOSIBLE/financial-sentiment", revision=revision("NOSIBLE/financial-sentiment"), split="train", streaming=True)
    cands, seen, n_stream, n_leak, n_dup = [], set(), 0, 0, 0
    for row in ds:
        n_stream += 1
        s = cut_snippet(row["text"])  # label column is never read
        if s is None:
            continue
        k = norm(s)
        if k in pb_norm:
            n_leak += 1
            continue
        if k in seen:
            n_dup += 1
            continue
        seen.add(k)
        cands.append(s)
        if len(cands) >= args.n_candidates:
            break
    print(
        f"\nstreamed rows={n_stream} candidates={len(cands)} leak_dropped={n_leak} "
        f"dup_dropped={n_dup} ({time.time() - t0:.0f}s)"
    )

    # --- 3. label, filter, balance ---
    t1 = time.time()
    probs = predict(tok, model, cands, device)
    conf, lab = probs.max(-1)
    print(f"teacher labelling {time.time() - t1:.0f}s")
    all_counts = {c: int((lab == i).sum()) for i, c in enumerate(CLASSES)}
    keep = conf >= args.conf
    kept = {
        c: [j for j in range(len(cands)) if keep[j] and lab[j] == i] for i, c in enumerate(CLASSES)
    }
    kept_counts = {c: len(v) for c, v in kept.items()}
    smallest = min(kept_counts.values())
    cap = min(int(smallest * args.max_ratio), args.max_total // 3 * 2)
    rng = random.Random(0)
    sel = {c: rng.sample(v, min(len(v), cap)) for c, v in kept.items()}
    # cap total
    total = sum(len(v) for v in sel.values())
    if total > args.max_total:
        f = args.max_total / total
        sel = {c: v[: int(len(v) * f)] for c, v in sel.items()}
    final = [{"text": cands[j], "label": CLASSES[i]} for i, c in enumerate(CLASSES) for j in sel[c]]
    rng.shuffle(final)
    final_counts = {c: len(sel[c]) for c in CLASSES}
    stats = {
        "streamed_rows": n_stream,
        "candidates": len(cands),
        "leak_dropped": n_leak,
        "dup_dropped": n_dup,
        "teacher_pred_counts_all": all_counts,
        "kept_conf>=%.2f" % args.conf: kept_counts,
        "final_counts": final_counts,
        "final_total": len(final),
    }
    print(json.dumps(stats, indent=2))
    json.dump(final, open(out / "pseudo_labeled.json", "w"))
    json.dump(stats, open(out / "label_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
