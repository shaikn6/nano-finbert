"""
Masked-token pretraining of the NanoFinBERT encoder on unlabeled finance text.

Corpus: 40,000 candidate sentences (same collection logic as round 2) + the 2,295
original TRAINING sentences. 2,000 candidate sentences are held out for validation.
The MLM head ties its output matrix to the token embedding; only the encoder
(embedding, encoder_layers, final_norm) is saved.

Usage: PYTHONPATH=src python scripts/pretrain_mlm.py --out-dir <work>/round3 --tok-path <tok.json>
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from distill_label import cut_snippet, fixed_split, norm  # noqa: E402

from finbert.model import NanoFinBERT  # noqa: E402
from finbert.tokenizer import FinancialTokenizer  # noqa: E402

N_SPECIAL = 5  # [PAD] [UNK] [CLS] [SEP] [MASK]
MASK_ID = 4


class MLMModel(nn.Module):
    def __init__(self, enc: NanoFinBERT, vocab: int):
        super().__init__()
        self.enc = enc
        h = enc.config["hidden_dim"]
        self.transform = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h))
        self.bias = nn.Parameter(torch.zeros(vocab))

    def forward(self, ids, mask):
        hs = self.enc(ids, mask)["last_hidden_state"]
        return self.transform(hs) @ self.enc.embedding.token_embedding.weight.T + self.bias


def collect(n_candidates, pb_norm):
    from datasets import load_dataset

    ds = load_dataset("NOSIBLE/financial-sentiment", split="train", streaming=True)
    cands, seen, n_stream, n_leak, n_dup = [], set(), 0, 0, 0
    for row in ds:
        n_stream += 1
        s = cut_snippet(row["text"])  # label column never read
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
        if len(cands) >= n_candidates:
            break
    print(f"streamed={n_stream} candidates={len(cands)} leak_dropped={n_leak} dup_dropped={n_dup}")
    return cands


def encode_all(tok, texts, max_len):
    ids, am = [], []
    for t in texts:
        e = tok.encode(t, max_length=max_len, padding=True, truncation=True)
        ids.append(e["input_ids"])
        am.append(e["attention_mask"])
    return torch.tensor(ids), torch.tensor(am)


def mask_batch(ids, am, vocab, gen):
    """BERT masking: 15% of non-special tokens; 80% [MASK], 10% random, 10% unchanged."""
    cand = (ids >= N_SPECIAL) & (am == 1)
    sel = (torch.rand(ids.shape, generator=gen) < 0.15) & cand
    labels = torch.where(sel, ids, torch.full_like(ids, -100))
    r = torch.rand(ids.shape, generator=gen)
    out = ids.clone()
    out[sel & (r < 0.8)] = MASK_ID
    rnd = sel & (r >= 0.8) & (r < 0.9)
    out[rnd] = torch.randint(N_SPECIAL, vocab, ids.shape, generator=gen)[rnd]
    return out, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/financial_phrasebank.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tok-path", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-candidates", type=int, default=40000)
    ap.add_argument("--n-heldout", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = json.load(open(args.data_path, encoding="utf-8"))
    train_idx, _, _ = fixed_split(len(samples))
    pb_norm = {norm(s["text"]) for s in samples}  # all 3,279 PhraseBank sentences
    corpus_file = out / "mlm_candidates.json"
    if corpus_file.exists():
        cands = json.load(open(corpus_file))
    else:
        cands = collect(args.n_candidates, pb_norm)
        json.dump(cands, open(corpus_file, "w"))
    assert not any(norm(c) in pb_norm for c in cands), "leak"
    orig = [samples[i]["text"] for i in train_idx]
    rng = random.Random(args.seed)
    perm = list(range(len(cands)))
    rng.shuffle(perm)
    held = [cands[i] for i in perm[: args.n_heldout]]
    train_texts = [cands[i] for i in perm[args.n_heldout :]] + orig
    print(
        f"pretrain train={len(train_texts)} (news {len(cands) - args.n_heldout} + orig {len(orig)}) heldout={len(held)}"
    )

    tok = FinancialTokenizer.load(args.tok_path)
    print("tokenizer:", tok)
    vocab = 8000
    tr_ids, tr_am = encode_all(tok, train_texts, args.max_length)
    ho_ids, ho_am = encode_all(tok, held, args.max_length)

    dev = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    enc = NanoFinBERT(vocab_size=vocab, hidden_dim=128, num_layers=4, num_heads=4, max_seq_len=256)
    print(f"encoder model params (incl. unused pooler/sentiment head): {enc.count_parameters():,}")
    model = MLMModel(enc, vocab).to(dev)

    decay = [
        p
        for n, p in model.named_parameters()
        if not any(k in n for k in ["bias", "norm"]) and "transform.2" not in n
    ]
    nodecay = [
        p
        for n, p in model.named_parameters()
        if any(k in n for k in ["bias", "norm"]) or "transform.2" in n
    ]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd}, {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr,
    )
    n = len(train_texts)
    spe = math.ceil(n / args.batch_size)
    total = spe * args.epochs
    warm = max(1, int(0.05 * total))

    def lam(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)

    @torch.no_grad()
    def evaluate():
        model.eval()
        g = torch.Generator().manual_seed(1234)  # fixed masks for validation
        ls, correct, cnt = 0.0, 0, 0
        for i in range(0, len(ho_ids), 256):
            ids, am = ho_ids[i : i + 256], ho_am[i : i + 256]
            x, y = mask_batch(ids, am, vocab, g)
            lg = model(x.to(dev), am.to(dev))
            y = y.to(dev)
            m = y != -100
            ls += F.cross_entropy(lg[m], y[m], reduction="sum").item()
            correct += (lg[m].argmax(-1) == y[m]).sum().item()
            cnt += m.sum().item()
        model.train()
        return ls / cnt, correct / cnt

    gen = torch.Generator().manual_seed(args.seed)
    hist = []
    t0 = time.time()
    model.train()
    for ep in range(1, args.epochs + 1):
        order = torch.randperm(n, generator=gen)
        tl, tn = 0.0, 0
        for i in range(0, n, args.batch_size):
            b = order[i : i + args.batch_size]
            ids, am = tr_ids[b], tr_am[b]
            x, y = mask_batch(ids, am, vocab, gen)
            x, am_d, y = x.to(dev), am.to(dev), y.to(dev)
            lg = model(x, am_d)
            loss = F.cross_entropy(lg.view(-1, vocab), y.view(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item()
            tn += 1
        vl, va = evaluate()
        hist.append(
            {
                "epoch": ep,
                "train_loss": tl / tn,
                "val_mlm_loss": vl,
                "val_mlm_acc": va,
                "secs": time.time() - t0,
            }
        )
        print(
            f"epoch {ep:2d}/{args.epochs} train_loss={tl / tn:.4f} val_mlm_loss={vl:.4f} val_mlm_acc={va:.4f} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    keep = ("embedding.", "encoder_layers.", "final_norm.")
    sd = {k: v.cpu() for k, v in enc.state_dict().items() if k.startswith(keep)}
    torch.save(
        {"state_dict": sd, "config": enc.config, "args": vars(args)}, out / "pretrained_encoder.pt"
    )
    json.dump(
        {
            "args": vars(args),
            "device": dev,
            "n_train": n,
            "n_heldout": len(held),
            "history": hist,
            "total_secs": time.time() - t0,
        },
        open(out / "pretrain_results.json", "w"),
        indent=2,
    )
    print(f"saved {len(sd)} tensors; total {time.time() - t0:.0f}s on {dev}")


if __name__ == "__main__":
    main()
