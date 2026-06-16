# nano-finbert

![CI](https://github.com/shaikn6/nano-finbert/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

**The simplest, most readable financial NLP model you can actually understand.**

nano-finbert is a tiny transformer encoder (≈2M parameters) trained **from scratch** on financial text — no pretrained weights, no HuggingFace dependency, no black boxes. Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), every component is annotated to explain *why* it exists.

Feed it a financial headline. Get back a structured market signal.

```python
from finbert.signals import SignalExtractor

extractor = SignalExtractor()
signal = extractor.extract(
    "SpaceX's long-anticipated IPO filing sent aerospace stocks soaring to record highs"
)
print(signal.to_dict())
```

```json
{
  "text": "SpaceX's long-anticipated IPO filing sent aerospace stocks soaring to record highs",
  "sentiment": "positive",
  "confidence": 0.82,
  "entities": ["SpaceX", "aerospace"],
  "sectors": ["aerospace", "equities"],
  "event_type": "ipo",
  "impact_score": 0.71,
  "signal_direction": "bullish"
}
```

---

## Architecture

```mermaid
graph LR
    A[Financial Text] --> B[FinancialTokenizer<br/>BPE, 8000 vocab]
    B --> C[FinancialEmbedding<br/>token + positional]
    C --> D[EncoderLayer × 4<br/>MHA + FFN + LN]
    D --> E[Pooler<br/>CLS token]
    E --> F[SentimentHead<br/>pos / neg / neutral]
    F --> G[SignalExtractor<br/>entities + sectors + events]
    G --> H[FinancialSignal JSON]
```

## Model Architecture

| Component | Spec |
|-----------|------|
| Layers | 4 encoder layers |
| Hidden dim | 128 |
| Attention heads | 4 |
| Max sequence length | 256 tokens |
| Vocabulary size | 8,000 |
| Total parameters | ~2M |
| Training device | CPU (no GPU needed) |

## Quick Start

### Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn pydantic
git clone https://github.com/shaikn6/nano-finbert
cd nano-finbert
export PYTHONPATH=src
```

### Extract a signal

```python
from finbert.signals import SignalExtractor

extractor = SignalExtractor()  # uses random weights; load a checkpoint for real inference

# Single signal
signal = extractor.extract("Bitcoin dropped below $60,000 as SEC regulatory pressure mounted")
print(signal)
# FinancialSignal(sentiment='negative', direction='bearish', confidence=0.74, impact=0.52, ...)

# Batch extraction
texts = [
    "Gold futures fell 2.3% amid dollar strength and Fed rate hike concerns",
    "NVIDIA quarterly revenue hit $22 billion on AI accelerator demand",
    "ECB held interest rates steady, signaling a data-dependent approach",
]
signals = extractor.extract_batch(texts)
for s in signals:
    print(f"[{s.signal_direction:7s}] [{s.event_type:15s}] {s.text[:60]}")
```

### Serve the API

```bash
docker-compose up
# or:
uvicorn finbert.api.server:app --host 0.0.0.0 --port 8000 --reload
```

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "SpaceX IPO raised $10 billion at a $250 billion valuation"}'
```

### Train from scratch

```bash
python scripts/train.py --epochs 5 --batch-size 32 --output checkpoints/
```

## Supported Signal Types

| Field | Values |
|-------|--------|
| `sentiment` | `positive`, `negative`, `neutral` |
| `signal_direction` | `bullish`, `bearish`, `neutral` |
| `event_type` | `ipo`, `earnings`, `rate_decision`, `merger`, `commodity_move`, `regulatory`, `layoffs`, `product_launch`, `general` |
| `sectors` | `tech`, `aerospace`, `commodities`, `crypto`, `forex`, `equities`, `fixed_income` |

## Training

nano-finbert uses a tiny educational dataset (250+ curated financial phrases) included in `data/samples/financial_phrases.json`. The training loop in `src/finbert/train.py` is heavily annotated to explain every decision.

**Expected training behavior (5 epochs on sample data):**
- Initial loss: ~1.1 (random baseline for 3-class classification)
- Loss at convergence: ~0.7–0.8 on training set
- Accuracy on training set: ~65–75%

For real-world accuracy, fine-tune on [Financial PhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) (4,846 labeled sentences).

## What's different from FinBERT / HuggingFace?

| | nano-finbert | FinBERT (HuggingFace) |
|---|---|---|
| Dependencies | PyTorch only | transformers, tokenizers, huggingface-hub |
| Model size | ~2M params | 110M params |
| Training | From scratch | Fine-tuned from BERT |
| Readability | Educational, annotated | Production library |
| Output | Structured `FinancialSignal` | Raw logits / token labels |
| Purpose | Learning + prototyping | Production NLP |

## Project Structure

```
nano-finbert/
├── src/finbert/
│   ├── model.py        # Transformer architecture (annotated)
│   ├── tokenizer.py    # BPE tokenizer for financial text
│   ├── dataset.py      # PyTorch Dataset wrapper
│   ├── train.py        # Training loop (educational)
│   ├── signals.py      # FinancialSignal extractor
│   └── api/server.py   # FastAPI inference endpoint
├── data/samples/       # 250+ labeled financial phrases
├── tests/              # pytest test suite (70%+ coverage)
└── scripts/            # train.py + infer.py CLIs
```

## License

MIT
