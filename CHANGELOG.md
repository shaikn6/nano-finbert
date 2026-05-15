# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-06-16

### Added
- Tiny transformer (~2M parameters) trained from scratch on financial news, earnings calls, and SEC filings
- Structured market signal output: sentiment polarity, named entities, sector tags, event type, and impact score
- Custom financial tokenizer with domain-specific vocabulary covering tickers, ratios, and regulatory terms
- Training pipeline with mixed-precision support, gradient checkpointing, and WandB experiment tracking
- Inference API returning JSON market signals with confidence intervals for downstream consumption
- Benchmark suite comparing nano-finbert accuracy against FinBERT-base on sentiment and NER tasks

### Changed
- Production-ready CI/CD with 95%+ test coverage enforcement

### Security
- Model weights and training data stored locally; no external telemetry during inference
