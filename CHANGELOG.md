# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added
- `scripts/eval_sentiment.py` and `data/eval/headlines_sentiment.json` (90 hand-written, class-balanced headlines): a committed command that scores the shipped sentiment head, 0.678 accuracy / 0.675 macro-F1
- Shipped trained checkpoint (`src/finbert/weights/`, 7.6 MB) trained by `scripts/train_tfns.py` on the MIT-licensed zeroshot/twitter-financial-news-sentiment dataset: 0.760 test accuracy, 0.680 macro-F1 (TF-IDF + logistic regression: 0.828 / 0.753). `SignalExtractor.pretrained()` loads it; the API and `scripts/infer.py` use it when no model path is given
- Distillation experiment: `scripts/distill_label.py`, `scripts/round2_teacher.py`, `scripts/pretrain_mlm.py` and `scripts/train_distill.py`, with measured results in the README (the same 1.88M-parameter model reaches 0.633 accuracy with teacher-labeled data and 0.654 with masked-token pretraining as well, against 0.552 for the original recipe)
- Tokenizer round-trip test asserting `decode(encode(text))` recovers the original character stream (BPE is lossless over characters), and a strengthened training-step test that asserts the loss actually falls while overfitting one fixed batch

### Changed
- README production-model numbers updated to the current v3 card (97.94% / macro-F1 0.9643 on the PhraseBank `Sentences_AllAgree` test split; v2 96.47%, v1 95.29%); a Limitations section; the PhraseBank and distillation figures are marked as console output of runs with no committed result file
- README now positions nano-finbert against the fine-tuned sibling repos (`fin-lora`, `finarena`): it is the only one with no pretrained weights and its own from-scratch BPE tokenizer
- `SignalExtractor` now exposes `trained` (False when no checkpoint is loaded) and logs a warning; the API adds `trained` and `warning` to `/extract`, `/extract/batch`, `/health` and `/model/info` so random-weight output is not mistaken for real predictions. `/health` `model_loaded` is unchanged (service is up)
- README no longer presents the production model's 95.29% and this model's 55.2% as scores on the same test set; the production model scores 74.3% on this repo's test split
- README production-model numbers now match its Hugging Face card: v2 scores 96.47% / macro-F1 0.9532 on the PhraseBank `Sentences_AllAgree` test split (v1: 95.29% / 0.937); the 74.3% figure and the distillation teacher are labelled v1, the revision pinned in `scripts/pins.py`. The SpaceX example output comes from a real run with the shipped checkpoint, and the test count is 433

### Fixed
- `SignalExtractor(model_path=...)` with no `tokenizer_path` (API: `MODEL_PATH` without `TOKENIZER_PATH`) silently paired the checkpoint with the default tokenizer and still reported `trained=True`. It now loads the `tokenizer.json` saved next to the checkpoint; if there is none it logs a warning and reports `trained=False`

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
