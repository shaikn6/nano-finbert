"""
CLI training script for NanoFinBERT.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 20 --lr 1e-4 --batch-size 32
    python scripts/train.py --data-path data/samples/financial_phrases.json --checkpoint-dir out/

All arguments are optional — defaults train on the built-in sample dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finbert.dataset import load_sample_dataset, make_dataloaders
from finbert.model import NanoFinBERT
from finbert.tokenizer import build_default_tokenizer
from finbert.train import TrainConfig, Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train NanoFinBERT from scratch on financial text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-path", default=None, help="Path to financial_phrases.json")
    p.add_argument(
        "--checkpoint-dir", default="checkpoints", help="Directory for saved checkpoints"
    )
    p.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    p.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    p.add_argument("--warmup-steps", type=int, default=100, help="LR warmup steps")
    p.add_argument("--vocab-size", type=int, default=8000, help="Tokenizer vocabulary size")
    p.add_argument("--hidden-dim", type=int, default=128, help="Transformer hidden dimension")
    p.add_argument("--num-layers", type=int, default=4, help="Number of encoder layers")
    p.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    p.add_argument("--max-seq-len", type=int, default=256, help="Maximum sequence length")
    p.add_argument("--seed", type=int, default=42, help="Random seed for data split")
    p.add_argument("--tokenizer-save", default=None, help="Path to save trained tokenizer")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("nano-finbert training script")
    print("=" * 60)

    # --- Build tokenizer ---
    print("\n[1/4] Building tokenizer...")
    data_path = args.data_path

    import json

    if data_path:
        with open(data_path, encoding="utf-8") as f:
            samples = json.load(f)
        texts = [s["text"] for s in samples]
    else:
        # Load built-in samples for tokenizer training
        default_path = Path(__file__).parent.parent / "data" / "samples" / "financial_phrases.json"
        with open(default_path, encoding="utf-8") as f:
            samples = json.load(f)
        texts = [s["text"] for s in samples]
        data_path = str(default_path)

    tokenizer = build_default_tokenizer(texts)
    print(f"    Tokenizer ready: {tokenizer}")

    if args.tokenizer_save:
        tokenizer.save(args.tokenizer_save)
        print(f"    Tokenizer saved to {args.tokenizer_save}")

    # --- Load dataset ---
    print("\n[2/4] Loading dataset...")
    dataset = load_sample_dataset(tokenizer, max_length=args.max_seq_len, data_path=data_path)
    dist = dataset.label_distribution()
    print(f"    Total examples: {len(dataset)}")
    print(f"    Label distribution: {dist}")

    train_loader, val_loader = make_dataloaders(
        dataset, train_ratio=0.8, batch_size=args.batch_size, seed=args.seed
    )
    print(f"    Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # --- Build model ---
    print("\n[3/4] Building model...")
    model = NanoFinBERT(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
    )
    print(f"    Parameters: {model.count_parameters():,}")
    breakdown = model.parameter_breakdown()
    for component, count in breakdown.items():
        print(f"      {component}: {count:,}")

    # --- Train ---
    print("\n[4/4] Training...")
    config = TrainConfig(
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_steps=200,
        eval_every_n_epochs=1,
        log_every_n_steps=10,
    )

    trainer = Trainer(model, tokenizer, train_loader, val_loader, config)
    history = trainer.train()

    print("\nTraining complete!")
    if history:
        last = history[-1]
        print(f"  Final train loss: {last.get('train_loss', 'N/A'):.4f}")
        if "accuracy" in last:
            print(f"  Final val accuracy: {last['accuracy']:.3f}")

    print(f"\nCheckpoints saved to: {args.checkpoint_dir}/")
    print("Best model: checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
