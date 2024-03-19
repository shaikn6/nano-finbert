"""
Signal extractor: converts raw financial text into structured trading signals.

This is the KEY differentiator of nano-finbert. Rather than returning raw
sentiment probabilities, we output a structured FinancialSignal that downstream
trading systems, alerting pipelines, or dashboards can consume directly.

Architecture:
  - NanoFinBERT model → sentiment + confidence
  - Rule-based entity tagger → companies, tickers, commodities
  - Rule-based sector classifier → tech / aerospace / crypto / forex / commodities
  - Rule-based event typer → ipo / earnings / rate_decision / merger / commodity_move
  - Impact scorer → 0.0–1.0 estimated market impact heuristic

The rule-based components work WITHOUT a trained model checkpoint,
making the extractor immediately useful for prototyping and testing.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F

from finbert.dataset import ID_TO_SENTIMENT
from finbert.model import NanoFinBERT
from finbert.tokenizer import FinancialTokenizer, build_default_tokenizer

# ---------------------------------------------------------------------------
# Sector keyword mapping
# ---------------------------------------------------------------------------

SECTOR_KEYWORDS: dict[str, list[str]] = {
    "tech": [
        "software",
        "ai",
        "chip",
        "semiconductor",
        "cloud",
        "saas",
        "startup",
        "tech",
        "silicon",
        "data center",
        "gpu",
        "cpu",
        "server",
        "cybersecurity",
        "machine learning",
        "artificial intelligence",
        "openai",
        "nvidia",
        "intel",
        "apple",
        "microsoft",
        "google",
        "meta",
        "alphabet",
        "amazon web services",
        "aws",
        "azure",
        "gcp",
    ],
    "aerospace": [
        "spacex",
        "rocket",
        "satellite",
        "launch",
        "nasa",
        "boeing",
        "lockheed",
        "airbus",
        "raytheon",
        "northrop",
        "defense",
        "spacecraft",
        "orbit",
        "starship",
        "falcon",
        "artemis",
        "iss",
        "drone",
        "aviation",
        "airline",
    ],
    "commodities": [
        "gold",
        "silver",
        "copper",
        "platinum",
        "palladium",
        "lithium",
        "cobalt",
        "oil",
        "crude",
        "brent",
        "wti",
        "natural gas",
        "lng",
        "petroleum",
        "wheat",
        "corn",
        "soybeans",
        "cotton",
        "sugar",
        "coffee",
        "cocoa",
        "iron ore",
        "coal",
        "uranium",
        "nickel",
        "zinc",
        "aluminum",
    ],
    "crypto": [
        "bitcoin",
        "ethereum",
        "crypto",
        "blockchain",
        "defi",
        "nft",
        "altcoin",
        "btc",
        "eth",
        "usdt",
        "bnb",
        "solana",
        "ripple",
        "xrp",
        "dogecoin",
        "stablecoin",
        "web3",
        "dao",
        "yield farming",
        "layer 2",
        "polygon",
        "binance",
        "coinbase",
        "kraken",
        "ftx",
    ],
    "forex": [
        "dollar",
        "euro",
        "yen",
        "pound",
        "yuan",
        "franc",
        "ruble",
        "won",
        "usd",
        "eur",
        "jpy",
        "gbp",
        "cny",
        "chf",
        "aud",
        "cad",
        "nzd",
        "fed",
        "ecb",
        "boj",
        "bank of england",
        "fomc",
        "currency",
        "fx",
        "exchange rate",
        "monetary policy",
        "interest rate",
        "rate hike",
        "rate cut",
        "quantitative easing",
        "qe",
        "taper",
        "inflation",
        "cpi",
    ],
    "equities": [
        "stock",
        "shares",
        "equity",
        "ipo",
        "listing",
        "s&p",
        "nasdaq",
        "dow",
        "russell",
        "earnings",
        "eps",
        "revenue",
        "guidance",
        "buyback",
        "dividend",
        "p/e",
        "market cap",
        "valuation",
        "rally",
        "selloff",
        "correction",
    ],
    "fixed_income": [
        "bond",
        "treasury",
        "yield",
        "10-year",
        "2-year",
        "spread",
        "credit",
        "investment grade",
        "high yield",
        "junk",
        "coupon",
        "duration",
        "sovereign debt",
        "municipal",
        "corporate bond",
        "maturity",
    ],
}

# ---------------------------------------------------------------------------
# Event type keyword mapping
# ---------------------------------------------------------------------------

EVENT_KEYWORDS: dict[str, list[str]] = {
    "ipo": [
        "ipo",
        "initial public offering",
        "listing",
        "debut",
        "went public",
        "s-1",
        "prospectus",
        "underwriter",
        "road show",
        "lock-up",
    ],
    "earnings": [
        "earnings",
        "quarterly results",
        "q1",
        "q2",
        "q3",
        "q4",
        "annual results",
        "eps",
        "revenue beat",
        "revenue miss",
        "guidance",
        "outlook",
        "forecast",
        "profit",
        "net income",
        "operating income",
        "ebitda",
    ],
    "rate_decision": [
        "rate hike",
        "rate cut",
        "interest rate",
        "basis points",
        "bps",
        "fomc",
        "federal reserve",
        "ecb",
        "bank of england",
        "monetary policy",
        "tapering",
        "quantitative easing",
        "qe",
        "hawkish",
        "dovish",
    ],
    "merger": [
        "acquisition",
        "merger",
        "takeover",
        "buyout",
        "deal",
        "agreement",
        "private equity",
        "pe",
        "lbo",
        "leveraged buyout",
        "all-cash",
        "all-stock",
        "bid",
        "offer",
        "premium",
    ],
    "commodity_move": [
        "fell",
        "rose",
        "jumped",
        "dropped",
        "surged",
        "plunged",
        "slipped",
        "gained",
        "lost",
        "climbed",
        "tumbled",
    ],
    "regulatory": [
        "sec",
        "ftc",
        "doj",
        "antitrust",
        "fine",
        "penalty",
        "lawsuit",
        "settlement",
        "probe",
        "investigation",
        "compliance",
        "regulation",
        "approval",
        "rejected",
        "ban",
        "sanction",
    ],
    "layoffs": [
        "layoffs",
        "job cuts",
        "restructuring",
        "downsizing",
        "reduction in force",
        "rif",
        "redundancies",
        "workforce reduction",
        "headcount",
    ],
    "product_launch": [
        "launch",
        "unveiled",
        "announced",
        "released",
        "debut",
        "new product",
        "partnership",
        "collaboration",
        "expansion",
        "enters",
    ],
}

# ---------------------------------------------------------------------------
# Known financial entities (for quick entity tagging)
# ---------------------------------------------------------------------------

KNOWN_ENTITIES: list[tuple[str, str]] = [
    # (pattern, entity_type)
    (r"\$[A-Z]{1,5}", "ticker"),
    (r"\bApple\b", "company"),
    (r"\bMicrosoft\b", "company"),
    (r"\bGoogle\b", "company"),
    (r"\bAlphabet\b", "company"),
    (r"\bAmazon\b", "company"),
    (r"\bMeta\b", "company"),
    (r"\bTesla\b", "company"),
    (r"\bNvidia\b", "company"),
    (r"\bSpaceX\b", "company"),
    (r"\bStripe\b", "company"),
    (r"\bKlarna\b", "company"),
    (r"\bOpenAI\b", "company"),
    (r"\bJPMorgan\b", "company"),
    (r"\bGoldman\b", "company"),
    (r"\bBlackRock\b", "company"),
    (r"\bBerkshire\b", "company"),
    (r"\bBitcoin\b|\bBTC\b", "crypto"),
    (r"\bEthereum\b|\bETH\b", "crypto"),
    (r"\bFed\b|\bFOMC\b|\bFederal Reserve\b", "institution"),
    (r"\bECB\b|\bEuropean Central Bank\b", "institution"),
    (r"\bgold\b|\bGold\b", "commodity"),
    (r"\bsilver\b|\bSilver\b", "commodity"),
    (r"\bcrude oil\b|\bWTI\b|\bBrent\b", "commodity"),
    (r"\bcopper\b|\bCopper\b", "commodity"),
    (r"\blithium\b|\bLithium\b", "commodity"),
    (r"\bUSD\b|\bdollar\b", "currency"),
    (r"\bEUR\b|\beuro\b", "currency"),
    (r"\bJPY\b|\byen\b", "currency"),
    (r"\bGBP\b|\bpound\b", "currency"),
]

# ---------------------------------------------------------------------------
# Impact score heuristics
# ---------------------------------------------------------------------------

HIGH_IMPACT_PATTERNS: list[str] = [
    r"\d+%",  # percentage moves
    r"billion|trillion",  # large dollar amounts
    r"record|historic|all-time",  # record-breaking events
    r"crash|crisis|collapse|default",  # severe negative events
    r"acquisition|merger|takeover",  # M&A
    r"ipo|initial public offering",  # IPOs
    r"rate hike|rate cut|fomc",  # central bank decisions
    r"bankruptcy|chapter 11",  # distress events
]

SENTIMENT_TO_DIRECTION: dict[str, str] = {
    "positive": "bullish",
    "negative": "bearish",
    "neutral": "neutral",
}


# ---------------------------------------------------------------------------
# FinancialSignal dataclass
# ---------------------------------------------------------------------------


@dataclass
class FinancialSignal:
    """
    Structured representation of financial intelligence extracted from text.

    Fields:
        text:             Original input text.
        sentiment:        "positive" | "negative" | "neutral"
        confidence:       Model confidence score (0.0–1.0).
        entities:         List of detected entities (companies, tickers, commodities).
        sectors:          List of relevant sectors (tech, crypto, forex, etc.).
        event_type:       Detected event category (ipo, earnings, merger, etc.).
        impact_score:     Estimated market impact magnitude (0.0–1.0).
        signal_direction: "bullish" | "bearish" | "neutral"
    """

    text: str
    sentiment: str
    confidence: float
    entities: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    event_type: str = "general"
    impact_score: float = 0.5
    signal_direction: str = "neutral"

    def to_dict(self) -> dict:
        """Return the signal as a plain dictionary (JSON-serialisable)."""
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"FinancialSignal("
            f"sentiment={self.sentiment!r}, "
            f"direction={self.signal_direction!r}, "
            f"confidence={self.confidence:.2f}, "
            f"impact={self.impact_score:.2f}, "
            f"event={self.event_type!r}, "
            f"sectors={self.sectors}, "
            f"entities={self.entities[:3]}{'...' if len(self.entities) > 3 else ''}"
            f")"
        )


# ---------------------------------------------------------------------------
# Rule-based helpers
# ---------------------------------------------------------------------------


def _extract_entities(text: str) -> list[str]:
    """Extract named entities using regex patterns against known entity list."""
    found: list[str] = []
    seen: set[str] = set()
    for pattern, _ in KNOWN_ENTITIES:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entity = match.group(0)
            if entity not in seen:
                found.append(entity)
                seen.add(entity)
    return found


def _detect_sectors(text: str) -> list[str]:
    """Detect which financial sectors are referenced in the text."""
    text_lower = text.lower()
    detected: list[str] = []
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            detected.append(sector)
    return detected or ["general"]


def _detect_event_type(text: str) -> str:
    """Identify the primary event type from the text."""
    text_lower = text.lower()
    for event_type, keywords in EVENT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return event_type
    return "general"


def _compute_impact_score(text: str, confidence: float) -> float:
    """
    Estimate market impact on a 0.0–1.0 scale.

    Heuristic: more high-impact patterns found → higher score.
    Weighted by model confidence.
    """
    text_lower = text.lower()
    hits = sum(1 for p in HIGH_IMPACT_PATTERNS if re.search(p, text_lower))
    base_score = min(hits / len(HIGH_IMPACT_PATTERNS) + 0.2, 1.0)
    # Blend with confidence: low-confidence signals get dampened impact
    return round(base_score * (0.5 + 0.5 * confidence), 3)


# ---------------------------------------------------------------------------
# Signal Extractor
# ---------------------------------------------------------------------------


class SignalExtractor:
    """
    Converts raw financial text into structured FinancialSignal objects.

    Combines:
      1. NanoFinBERT model for sentiment classification (with confidence)
      2. Rule-based entity extraction (patterns → company/ticker/commodity names)
      3. Rule-based sector detection (keyword matching)
      4. Rule-based event typing (IPO/earnings/M&A/etc.)
      5. Impact scoring heuristic

    The extractor works in two modes:
      - With a trained model: uses real sentiment probabilities.
      - Without a model (model_path=None): uses random-weight model (for testing).

    Usage:
        extractor = SignalExtractor()  # uses random weights — fine for testing
        signal = extractor.extract("SpaceX IPO filing sent aerospace stocks soaring")
        print(signal.to_dict())
    """

    def __init__(
        self,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = device

        # Build tokenizer
        if tokenizer_path:
            self.tokenizer = FinancialTokenizer.load(tokenizer_path)
        else:
            # Use default tokenizer (trained on seed vocabulary)
            self.tokenizer = build_default_tokenizer()

        # Build model
        if model_path:
            self.model = NanoFinBERT.from_checkpoint(model_path)
        else:
            # Random weights — valid for structural testing, not for real inference
            self.model = NanoFinBERT()

        self.model.to(device)
        self.model.eval()

    def extract(self, text: str) -> FinancialSignal:
        """
        Extract a structured financial signal from a single text string.

        Args:
            text: Raw financial news headline or sentence.

        Returns:
            FinancialSignal with all fields populated.
        """
        # --- Sentiment via model ---
        encoded = self.tokenizer.encode(text, max_length=256, padding=True)
        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs["sentiment_logits"]  # (1, 3)
            probs = F.softmax(logits, dim=-1).squeeze(0)  # (3,)

        predicted_id = probs.argmax().item()
        confidence = probs[predicted_id].item()
        sentiment = ID_TO_SENTIMENT[predicted_id]

        # --- Rule-based extraction ---
        entities = _extract_entities(text)
        sectors = _detect_sectors(text)
        event_type = _detect_event_type(text)
        impact_score = _compute_impact_score(text, confidence)
        signal_direction = SENTIMENT_TO_DIRECTION[sentiment]

        return FinancialSignal(
            text=text,
            sentiment=sentiment,
            confidence=round(confidence, 4),
            entities=entities,
            sectors=sectors,
            event_type=event_type,
            impact_score=impact_score,
            signal_direction=signal_direction,
        )

    def extract_batch(self, texts: list[str]) -> list[FinancialSignal]:
        """
        Extract signals from a list of texts.

        Processes in a single batched forward pass for efficiency.

        Args:
            texts: List of financial text strings.

        Returns:
            List of FinancialSignal objects in the same order as input.
        """
        if not texts:
            return []

        # Encode all texts
        encodings = [self.tokenizer.encode(t, max_length=256, padding=True) for t in texts]
        input_ids = torch.tensor([e["input_ids"] for e in encodings], dtype=torch.long).to(
            self.device
        )
        attention_mask = torch.tensor(
            [e["attention_mask"] for e in encodings], dtype=torch.long
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs["sentiment_logits"]  # (batch, 3)
            probs = F.softmax(logits, dim=-1)  # (batch, 3)

        signals: list[FinancialSignal] = []
        for i, text in enumerate(texts):
            predicted_id = probs[i].argmax().item()
            confidence = probs[i][predicted_id].item()
            sentiment = ID_TO_SENTIMENT[predicted_id]

            entities = _extract_entities(text)
            sectors = _detect_sectors(text)
            event_type = _detect_event_type(text)
            impact_score = _compute_impact_score(text, confidence)
            signal_direction = SENTIMENT_TO_DIRECTION[sentiment]

            signals.append(
                FinancialSignal(
                    text=text,
                    sentiment=sentiment,
                    confidence=round(confidence, 4),
                    entities=entities,
                    sectors=sectors,
                    event_type=event_type,
                    impact_score=impact_score,
                    signal_direction=signal_direction,
                )
            )

        return signals
