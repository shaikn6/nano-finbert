# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added
- Distillation experiment: `scripts/distill_label.py`, `scripts/round2_teacher.py` and `scripts/train_distill.py`, with measured results in the README (the same 1.88M-parameter model reaches 0.633 accuracy with teacher-labeled data, against 0.552 for the original recipe)

### Changed
- README no longer presents the production model's 95.29% and this model's 55.2% as scores on the same test set; the production model scores 74.3% on this repo's test split

## [1.0.0] - 2026-06-16

### Added
- Tiny transformer (~2M parameters) trained from scratch on a curated set of 198 synthetic financial phrases (`data/samples/financial_phrases.json`)
- Structured market signal output: sentiment polarity, named entities, sector tags, event type, and impact score
- Custom financial tokenizer with domain-specific vocabulary covering tickers, ratios, and regulatory terms
- Training pipeline for the from-scratch transformer
- Inference API returning JSON market signals with a single confidence score for downstream consumption

### Changed
- Production-ready CI/CD with 95%+ test coverage enforcement

### Security
- Model weights and training data stored locally; no external telemetry during inference
