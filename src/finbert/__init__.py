"""
nano-finbert: Educational tiny transformer for financial signal extraction.

Trained from scratch (no pretrained weights), ~2M parameters, CPU-friendly.
Inspired by nanoGPT — simple, readable, educational.
"""

__version__ = "0.1.0"
__author__ = "nano-finbert contributors"

from finbert.model import NanoFinBERT
from finbert.signals import FinancialSignal, SignalExtractor
from finbert.tokenizer import FinancialTokenizer

__all__ = ["NanoFinBERT", "FinancialTokenizer", "SignalExtractor", "FinancialSignal"]
