"""
CLI inference script for NanoFinBERT.

Usage:
    # Single text (random-weight model, for testing)
    python scripts/infer.py --text "SpaceX IPO filing sent aerospace stocks soaring"

    # With trained checkpoint
    python scripts/infer.py --model checkpoints/best_model.pt --text "Gold fell 2%"

    # Batch from file (one text per line)
    python scripts/infer.py --model checkpoints/best_model.pt --input texts.txt

    # JSON output
    python scripts/infer.py --text "Bitcoin fell below 60k" --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finbert.signals import SignalExtractor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract financial signals using NanoFinBERT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--text", default=None, help="Single text to analyse")
    p.add_argument("--input", default=None, help="Path to a text file (one sentence per line)")
    p.add_argument("--model", default=None, help="Path to trained model checkpoint (.pt)")
    p.add_argument("--tokenizer", default=None, help="Path to saved tokenizer (.json)")
    p.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    p.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    return p.parse_args()


def print_signal(signal, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps(signal.to_dict(), indent=2))
        return

    direction_emoji = {"bullish": "+", "bearish": "-", "neutral": "~"}
    marker = direction_emoji.get(signal.signal_direction, "?")

    print(f"\n{'─' * 60}")
    print(f"  Text:      {signal.text[:80]}{'...' if len(signal.text) > 80 else ''}")
    print(f"  Sentiment: {signal.sentiment.upper()} ({signal.confidence:.0%} confidence)")
    print(f"  Direction: [{marker}] {signal.signal_direction.upper()}")
    print(f"  Impact:    {signal.impact_score:.2f} / 1.0")
    print(f"  Event:     {signal.event_type}")
    print(f"  Sectors:   {', '.join(signal.sectors)}")
    if signal.entities:
        print(f"  Entities:  {', '.join(signal.entities[:8])}")
    print(f"{'─' * 60}")


def main() -> None:
    args = parse_args()

    if not args.text and not args.input:
        print("Error: provide --text or --input", file=sys.stderr)
        sys.exit(1)

    print("Loading model...", file=sys.stderr)
    extractor = SignalExtractor(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )
    params = extractor.model.count_parameters()
    print(f"Model ready ({params:,} parameters)", file=sys.stderr)

    if args.text:
        signal = extractor.extract(args.text)
        print_signal(signal, args.json_output)

    elif args.input:
        path = Path(args.input)
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)

        texts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        print(f"Processing {len(texts)} texts...", file=sys.stderr)

        signals = extractor.extract_batch(texts)

        if args.json_output:
            print(json.dumps([s.to_dict() for s in signals], indent=2))
        else:
            for signal in signals:
                print_signal(signal, json_output=False)

        # Summary
        if not args.json_output:
            sentiments = [s.sentiment for s in signals]
            print(f"\nSummary: {len(signals)} signals extracted")
            print(f"  Positive: {sentiments.count('positive')}")
            print(f"  Negative: {sentiments.count('negative')}")
            print(f"  Neutral:  {sentiments.count('neutral')}")


if __name__ == "__main__":
    main()
