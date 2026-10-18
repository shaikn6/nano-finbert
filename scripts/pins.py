"""Hugging Face revisions pinned for reproducibility.

Each revision was the head of its repository on 2026-10-11. Neither repository had changed since
2026-06-24, before the distillation results in the README were produced, so these are the
versions behind those numbers.

To move a pin, change the hash here and rerun whatever depends on it.
"""

REVISIONS = {
    "9mark9/finbert-minilm-sentiment": "fdbfec0cd09610bd5af26da8998507fe7838e838",
    "NOSIBLE/financial-sentiment": "9bd8206a729ccd64fa259f0368f0b4025bb472af",
    # MIT-licensed; pinned 2026-10-17 for scripts/train_tfns.py
    "zeroshot/twitter-financial-news-sentiment": "ccbe24de388e287beb92dd393a335c376b350ac3",
}


def revision(repo_id: str) -> str | None:
    """Pinned revision for ``repo_id``; None (the library default) for a local path or an unlisted id."""
    return REVISIONS.get(repo_id)
