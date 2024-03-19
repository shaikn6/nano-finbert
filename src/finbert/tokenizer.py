"""
FinancialTokenizer — A simple BPE tokenizer trained on financial vocabulary.

Design philosophy:
  - Small vocabulary (8000 tokens) optimised for financial text
  - Special tokens for BERT-style masked language modelling
  - Built-in financial vocabulary seed (tickers, commodity names, FX pairs)
  - Pure Python implementation — no external tokenizer libraries required

BPE (Byte-Pair Encoding) algorithm summary:
  1. Start with a character-level vocabulary
  2. Count the most frequent adjacent pair of symbols
  3. Merge that pair into a new symbol
  4. Repeat until vocab_size is reached

This produces sub-word tokens that handle unseen words like "$NVDA" or
"hyperinflationary" by decomposing them into known sub-units.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Financial vocabulary seeds — ensures common terms are not split
# ---------------------------------------------------------------------------

FINANCIAL_SEED_VOCAB: list[str] = [
    # Ticker patterns
    "$AAPL",
    "$MSFT",
    "$NVDA",
    "$TSLA",
    "$GOOGL",
    "$AMZN",
    "$META",
    "$BRK",
    "$JPM",
    "$BAC",
    "$WMT",
    "$V",
    "$MA",
    "$UNH",
    "$XOM",
    "$CVX",
    "$HD",
    # Crypto
    "bitcoin",
    "ethereum",
    "crypto",
    "blockchain",
    "defi",
    "nft",
    "altcoin",
    "BTC",
    "ETH",
    "USDT",
    "BNB",
    "SOL",
    "XRP",
    # Commodities
    "gold",
    "silver",
    "copper",
    "platinum",
    "palladium",
    "lithium",
    "oil",
    "crude",
    "brent",
    "WTI",
    "natural gas",
    "wheat",
    "corn",
    "soybeans",
    "cotton",
    "iron ore",
    "coal",
    "uranium",
    # Currencies / Forex
    "dollar",
    "euro",
    "yen",
    "pound",
    "yuan",
    "franc",
    "ruble",
    "USD",
    "EUR",
    "JPY",
    "GBP",
    "CNY",
    "CHF",
    "AUD",
    "CAD",
    # Financial terms
    "earnings",
    "revenue",
    "profit",
    "loss",
    "dividend",
    "buyback",
    "IPO",
    "acquisition",
    "merger",
    "spinoff",
    "bankruptcy",
    "default",
    "inflation",
    "deflation",
    "recession",
    "GDP",
    "CPI",
    "PPI",
    "interest rate",
    "Fed",
    "ECB",
    "FOMC",
    "quantitative easing",
    "bull",
    "bear",
    "rally",
    "selloff",
    "correction",
    "crash",
    "volatility",
    "liquidity",
    "leverage",
    "margin",
    "short",
    "long",
    "futures",
    "options",
    "derivatives",
    "hedge",
    "arbitrage",
    "market cap",
    "P/E",
    "EPS",
    "EBITDA",
    "ROE",
    "ROI",
    # Sectors
    "tech",
    "healthcare",
    "financials",
    "energy",
    "utilities",
    "consumer",
    "industrial",
    "aerospace",
    "semiconductor",
    "biotech",
    # Companies
    "Apple",
    "Microsoft",
    "Google",
    "Amazon",
    "Meta",
    "Tesla",
    "Nvidia",
    "SpaceX",
    "Stripe",
    "Klarna",
    "OpenAI",
    "JPMorgan",
    "Goldman",
    "BlackRock",
    "Berkshire",
    # Events
    "acquisition",
    "IPO",
    "earnings",
    "guidance",
    "outlook",
    "downgrade",
    "upgrade",
    "lawsuit",
    "regulatory",
    "approval",
    "launch",
    "partnership",
    "layoffs",
    "expansion",
]


class FinancialTokenizer:
    """
    A BPE tokenizer pre-seeded with financial vocabulary.

    Special token IDs (reserved, do not change):
        [PAD]  = 0  — padding token (ignored by attention)
        [UNK]  = 1  — unknown token (out-of-vocabulary)
        [CLS]  = 2  — classification token (prepended to every sequence)
        [SEP]  = 3  — separator token (appended to every sequence)
        [MASK] = 4  — masking token (used in masked language modelling)

    Usage:
        tokenizer = FinancialTokenizer(vocab_size=8000)
        tokenizer.train(texts)
        encoded = tokenizer.encode("Gold fell 2% on dollar strength")
        print(encoded["input_ids"])   # [2, 45, 67, 89, 102, 3]
        print(tokenizer.decode(encoded["input_ids"]))
    """

    SPECIAL_TOKENS: dict[str, int] = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
    }

    def __init__(self, vocab_size: int = 8000) -> None:
        self.vocab_size = vocab_size

        # Token → ID mapping
        self.token_to_id: dict[str, int] = dict(self.SPECIAL_TOKENS)
        # ID → Token mapping (reverse)
        self.id_to_token: dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}

        # BPE merge rules: pairs to merge and their resulting token
        self.merges: dict[tuple[str, str], str] = {}

        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, texts: list[str]) -> None:
        """
        Train the BPE tokenizer on a list of financial texts.

        Steps:
          1. Pre-tokenise into words (whitespace + punctuation split)
          2. Represent each word as a sequence of characters + end marker
          3. Count word frequencies
          4. Iteratively merge the most frequent adjacent pair
          5. Add financial seed vocabulary to ensure coverage of key terms

        Args:
            texts: List of training sentences / documents.
        """
        # --- Pre-tokenise ---
        word_freqs: Counter[str] = Counter()
        for text in texts:
            words = self._pre_tokenize(text)
            word_freqs.update(words)

        # Represent each word as space-separated characters + </w> end marker
        # e.g., "gold" → "g o l d </w>"
        vocab: dict[str, int] = {
            " ".join(list(word) + ["</w>"]): freq for word, freq in word_freqs.items()
        }

        # Initialise token set with all unique characters found
        char_set: set[str] = set()
        for word in vocab:
            char_set.update(word.split())

        next_id = len(self.SPECIAL_TOKENS)

        # Add base characters to vocabulary
        for char in sorted(char_set):
            if char not in self.token_to_id:
                self.token_to_id[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1

        # Add financial seed vocabulary directly (bypasses BPE for key terms)
        for term in FINANCIAL_SEED_VOCAB:
            if term not in self.token_to_id and next_id < self.vocab_size - 1000:
                self.token_to_id[term] = next_id
                self.id_to_token[next_id] = term
                next_id += 1

        # --- BPE merge loop ---
        num_merges = self.vocab_size - next_id
        for _ in range(max(0, num_merges)):
            pairs = self._get_pair_frequencies(vocab)
            if not pairs:
                break

            # Find the most frequent pair
            best_pair = max(pairs, key=lambda p: pairs[p])
            best_freq = pairs[best_pair]
            if best_freq < 2:
                break

            # Merge the pair
            merged = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged
            vocab = self._merge_pair(best_pair, vocab)

            if merged not in self.token_to_id:
                self.token_to_id[merged] = next_id
                self.id_to_token[next_id] = merged
                next_id += 1

            if next_id >= self.vocab_size:
                break

        self._trained = True

    @staticmethod
    def _pre_tokenize(text: str) -> list[str]:
        """Split text into words, handling financial symbols and punctuation."""
        # Keep $TICKER patterns together, split on whitespace/punctuation otherwise
        text = text.lower()
        # Match $TICKERS, numbers with % or $, and regular words
        tokens = re.findall(r"\$[a-z]+|\d+\.?\d*[%$]?|[a-z]+", text)
        return [t for t in tokens if t]

    @staticmethod
    def _get_pair_frequencies(vocab: dict[str, int]) -> Counter:
        """Count frequency of all adjacent symbol pairs across the vocabulary."""
        pairs: Counter = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_pair(pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
        """Apply a merge rule to the vocabulary."""
        new_vocab: dict[str, int] = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word, freq in vocab.items():
            new_word = pattern.sub("".join(pair), word)
            new_vocab[new_word] = freq
        return new_vocab

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def _tokenize_word(self, word: str) -> list[str]:
        """Apply BPE merges to a single word and return sub-word tokens."""
        if not word:
            return []

        # Check if whole word is in vocabulary (handles seed vocab and common words)
        if word in self.token_to_id:
            return [word]

        # Start as characters + end marker
        symbols = list(word) + ["</w>"]

        # Apply merges greedily (longest match first via merge order)
        changed = True
        while changed and len(symbols) > 1:
            changed = False
            i = 0
            new_symbols = []
            while i < len(symbols) - 1:
                pair = (symbols[i], symbols[i + 1])
                if pair in self.merges:
                    new_symbols.append(self.merges[pair])
                    i += 2
                    changed = True
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            if i < len(symbols):
                new_symbols.append(symbols[i])
            symbols = new_symbols

        return symbols

    def encode(
        self,
        text: str,
        max_length: int = 256,
        padding: bool = True,
        truncation: bool = True,
    ) -> dict[str, list[int]]:
        """
        Encode text into model inputs.

        Format: [CLS] token1 token2 ... tokenN [SEP] [PAD] [PAD] ...

        Args:
            text:        Input text string.
            max_length:  Maximum sequence length (default 256).
            padding:     Pad to max_length with [PAD] tokens.
            truncation:  Truncate sequences longer than max_length.

        Returns:
            dict with:
                "input_ids":      list of token IDs
                "attention_mask": list of 1s (real) and 0s (padding)
        """
        words = self._pre_tokenize(text)
        tokens: list[str] = []

        for word in words:
            tokens.extend(self._tokenize_word(word))

        # Convert tokens to IDs (use [UNK] for unseen tokens)
        token_ids = [self.token_to_id.get(tok, self.SPECIAL_TOKENS["[UNK]"]) for tok in tokens]

        # Account for [CLS] and [SEP]
        max_content = max_length - 2

        if truncation and len(token_ids) > max_content:
            token_ids = token_ids[:max_content]

        # Prepend [CLS] and append [SEP]
        input_ids = [self.SPECIAL_TOKENS["[CLS]"]] + token_ids + [self.SPECIAL_TOKENS["[SEP]"]]
        attention_mask = [1] * len(input_ids)

        # Pad to max_length
        if padding:
            pad_length = max_length - len(input_ids)
            input_ids = input_ids + [self.SPECIAL_TOKENS["[PAD]"]] * pad_length
            attention_mask = attention_mask + [0] * pad_length

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs back to text.

        Args:
            ids:                  List of integer token IDs.
            skip_special_tokens:  If True, omit [CLS], [SEP], [PAD], [MASK].

        Returns:
            Decoded text string.
        """
        special_ids = set(self.SPECIAL_TOKENS.values()) if skip_special_tokens else set()

        tokens = []
        for id_ in ids:
            if id_ in special_ids:
                continue
            token = self.id_to_token.get(id_, "[UNK]")
            tokens.append(token)

        # Reconstruct text: remove BPE end markers, join sub-words
        text = " ".join(tokens)
        text = text.replace("</w> ", " ").replace("</w>", "")
        return text.strip()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save tokenizer state to a JSON file."""
        state = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "merges": [[list(k), v] for k, v in self.merges.items()],
            "trained": self._trained,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> FinancialTokenizer:
        """Load tokenizer from a JSON file."""
        with open(path, encoding="utf-8") as f:
            state = json.load(f)

        tokenizer = cls(vocab_size=state["vocab_size"])
        tokenizer.token_to_id = state["token_to_id"]
        tokenizer.id_to_token = {int(v): k for k, v in state["token_to_id"].items()}
        tokenizer.merges = {(tuple(k)[0], tuple(k)[1]): v for k, v in state["merges"]}
        tokenizer._trained = state.get("trained", True)
        return tokenizer

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def vocab_size_actual(self) -> int:
        """Actual size of the learned vocabulary."""
        return len(self.token_to_id)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __repr__(self) -> str:
        status = "trained" if self._trained else "untrained"
        return (
            f"FinancialTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"actual={self.vocab_size_actual}, "
            f"status={status})"
        )


def build_default_tokenizer(texts: list[str] | None = None) -> FinancialTokenizer:
    """
    Build and train a tokenizer on the provided texts.
    If no texts given, trains on a minimal financial corpus using the seed vocabulary.
    """
    tokenizer = FinancialTokenizer(vocab_size=8000)

    if texts is None:
        # Use seed vocabulary terms as minimal training corpus
        texts = [
            " ".join(FINANCIAL_SEED_VOCAB[i : i + 20])
            for i in range(0, len(FINANCIAL_SEED_VOCAB), 20)
        ]

    tokenizer.train(texts)
    return tokenizer


# ---------------------------------------------------------------------------
# Vocabulary statistics helper
# ---------------------------------------------------------------------------


def vocab_stats(tokenizer: FinancialTokenizer) -> dict:
    """Return statistics about the tokenizer vocabulary."""
    special = set(FinancialTokenizer.SPECIAL_TOKENS.keys())
    all_tokens = set(tokenizer.token_to_id.keys())
    regular_tokens = all_tokens - special

    multi_char = [t for t in regular_tokens if len(t) > 1]
    single_char = [t for t in regular_tokens if len(t) == 1]

    return {
        "total_tokens": len(all_tokens),
        "special_tokens": len(special),
        "multi_char_tokens": len(multi_char),
        "single_char_tokens": len(single_char),
        "coverage_examples": [t for t in list(regular_tokens)[:10]],
    }
