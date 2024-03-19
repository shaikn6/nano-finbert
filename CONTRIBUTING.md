# Contributing to nano-finbert

## Setup

```bash
git clone https://github.com/shaikn6/nano-finbert
cd nano-finbert
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Code Style

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Pull Request Process

1. Fork the repo and create a feature branch
2. Write tests first (TDD)
3. Ensure all tests pass and coverage stays above 70%
4. Run `ruff check` and `ruff format`
5. Open a PR with a clear description

## Areas for Contribution

- Expanding `data/samples/financial_phrases.json` with more examples
- Improving the BPE tokenizer vocabulary
- Adding more failure pattern detection
- Training the model on open financial datasets
- Adding multilingual support
