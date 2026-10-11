"""
Round 2 parts A and B.
A. Fine-tune the teacher on the 2,295 training sentences; pick epoch by validation accuracy; one test eval.
B. Re-collect the same candidates as round 1 (same logic), label with the fine-tuned teacher,
   keep top prob >= 0.90, balance (max ratio 1.5, total cap 20000), save probability vectors.

Usage: PYTHONPATH=src python scripts/round2_teacher.py --out-dir <work>/round2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from distill_label import CLASSES, TEACHER, cut_snippet, fixed_split, norm, predict  # noqa: E402

from finbert.metrics import classification_report, format_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/financial_phrasebank.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
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
    train_idx, val_idx, test_idx = fixed_split(len(samples))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(TEACHER)
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER).to(device)
    print("device:", device, "id2label:", model.config.id2label)

    def texts_labels(idx):
        return [samples[i]["text"] for i in idx], [
            CLASSES.index(samples[i]["sentiment"]) for i in idx
        ]

    tr_x, tr_y = texts_labels(train_idx)
    va_x, va_y = texts_labels(val_idx)
    te_x, te_y = texts_labels(test_idx)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    best_acc, best_ep, best_state, hist = -1.0, 0, None, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tr_x), generator=gen).tolist()
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            b = perm[i : i + args.batch_size]
            enc = tok(
                [tr_x[j] for j in b],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            y = torch.tensor([tr_y[j] for j in b], device=device)
            loss = torch.nn.functional.cross_entropy(model(**enc).logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        model.eval()
        vp = predict(tok, model, va_x, device).argmax(-1)
        vacc = float((vp == torch.tensor(va_y)).float().mean())
        hist.append({"epoch": ep, "train_loss": tot / nb, "val_acc": vacc})
        print(
            f"epoch {ep} train_loss={tot / nb:.4f} val_acc={vacc:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if vacc > best_acc:
            best_acc, best_ep = vacc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    model.save_pretrained(out / "teacher_ft")
    tok.save_pretrained(out / "teacher_ft")
    print(f"selected epoch {best_ep} (val acc {best_acc:.4f}); teacher saved")

    pred = predict(tok, model, te_x, device).argmax(-1)
    rep = classification_report(pred, torch.tensor(te_y), 3, CLASSES)
    print("\nFINE-TUNED TEACHER test (n=%d)" % len(te_x))
    print(format_report(rep))
    json.dump(
        {
            "best_epoch": best_ep,
            "best_val_acc": best_acc,
            "history": hist,
            "test": rep,
            "args": vars(args),
            "train_seconds": time.time() - t0,
        },
        open(out / "teacher_ft_report.json", "w"),
        indent=2,
        default=float,
    )

    # ---- B: re-collect candidates (same logic as distill_label.py) ----
    pb_norm = {norm(s["text"]) for s in samples}
    ds = load_dataset("NOSIBLE/financial-sentiment", split="train", streaming=True)
    cands, seen, n_stream, n_leak, n_dup = [], set(), 0, 0, 0
    for row in ds:
        n_stream += 1
        s = cut_snippet(row["text"])
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
    r1 = {p["text"] for p in json.load(open(out.parent / "pseudo_labeled.json"))}
    print(
        f"streamed={n_stream} candidates={len(cands)} leak={n_leak} dup={n_dup}; "
        f"round-1 kept texts found in candidates: {len(r1 & set(cands))}/{len(r1)}"
    )

    probs = predict(tok, model, cands, device)
    conf, lab = probs.max(-1)
    all_counts = {c: int((lab == i).sum()) for i, c in enumerate(CLASSES)}
    kept = {
        c: [j for j in range(len(cands)) if conf[j] >= args.conf and lab[j] == i]
        for i, c in enumerate(CLASSES)
    }
    kept_counts = {c: len(v) for c, v in kept.items()}
    cap = min(int(min(kept_counts.values()) * args.max_ratio), args.max_total // 3 * 2)
    rng = random.Random(0)
    sel = {c: rng.sample(v, min(len(v), cap)) for c, v in kept.items()}
    total = sum(len(v) for v in sel.values())
    if total > args.max_total:
        f = args.max_total / total
        sel = {c: v[: int(len(v) * f)] for c, v in sel.items()}
    final = [
        {"text": cands[j], "label": CLASSES[i], "probs": [round(float(x), 5) for x in probs[j]]}
        for i, c in enumerate(CLASSES)
        for j in sel[c]
    ]
    rng.shuffle(final)
    stats = {
        "streamed_rows": n_stream,
        "candidates": len(cands),
        "leak_dropped": n_leak,
        "dup_dropped": n_dup,
        "teacher_pred_counts_all": all_counts,
        f"kept_conf>={args.conf}": kept_counts,
        "final_counts": {c: len(sel[c]) for c in CLASSES},
        "final_total": len(final),
    }
    print(json.dumps(stats, indent=2))
    json.dump(final, open(out / "pseudo_labeled_r2.json", "w"))
    json.dump(stats, open(out / "label_stats_r2.json", "w"), indent=2)


if __name__ == "__main__":
    main()
