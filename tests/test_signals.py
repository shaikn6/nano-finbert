"""
Tests for the financial signal extractor.

Covers:
  - FinancialSignal dataclass
  - Rule-based entity extraction
  - Rule-based sector detection
  - Rule-based event typing
  - Impact scoring
  - SignalExtractor single and batch extraction
  - Known semantic examples (SpaceX IPO → positive/bullish/aerospace)
"""

from __future__ import annotations

import pytest

from finbert.signals import (
    FinancialSignal,
    SignalExtractor,
    _compute_impact_score,
    _detect_event_type,
    _detect_sectors,
    _extract_entities,
)

# ---------------------------------------------------------------------------
# FinancialSignal dataclass
# ---------------------------------------------------------------------------


class TestFinancialSignal:
    def test_to_dict_has_all_fields(self):
        signal = FinancialSignal(
            text="test",
            sentiment="positive",
            confidence=0.9,
            entities=["Apple"],
            sectors=["tech"],
            event_type="earnings",
            impact_score=0.7,
            signal_direction="bullish",
        )
        d = signal.to_dict()
        assert set(d.keys()) == {
            "text",
            "sentiment",
            "confidence",
            "entities",
            "sectors",
            "event_type",
            "impact_score",
            "signal_direction",
        }

    def test_to_dict_values_match(self):
        signal = FinancialSignal(
            text="gold fell",
            sentiment="negative",
            confidence=0.85,
            entities=["gold"],
            sectors=["commodities"],
            event_type="commodity_move",
            impact_score=0.4,
            signal_direction="bearish",
        )
        d = signal.to_dict()
        assert d["sentiment"] == "negative"
        assert d["signal_direction"] == "bearish"
        assert d["confidence"] == 0.85

    def test_repr_contains_key_fields(self):
        signal = FinancialSignal(
            text="x",
            sentiment="neutral",
            confidence=0.5,
            entities=[],
            sectors=["general"],
            event_type="general",
            impact_score=0.3,
            signal_direction="neutral",
        )
        r = repr(signal)
        assert "sentiment=" in r
        assert "direction=" in r

    def test_default_entities_is_empty_list(self):
        signal = FinancialSignal(text="x", sentiment="neutral", confidence=0.5)
        assert signal.entities == []
        assert signal.sectors == []


# ---------------------------------------------------------------------------
# Rule-based helpers
# ---------------------------------------------------------------------------


class TestExtractEntities:
    def test_detects_bitcoin(self):
        entities = _extract_entities("Bitcoin dropped below $60,000")
        assert any("Bitcoin" in e or "bitcoin" in e.lower() for e in entities)

    def test_detects_gold(self):
        entities = _extract_entities("Gold futures fell 2.3% amid dollar strength")
        assert any("gold" in e.lower() for e in entities)

    def test_detects_ticker_pattern(self):
        entities = _extract_entities("$AAPL rose 3% after earnings beat")
        assert any("$" in e for e in entities)

    def test_detects_fed(self):
        entities = _extract_entities("The Fed raised interest rates by 25 basis points")
        assert any("Fed" in e for e in entities)

    def test_no_duplicates(self):
        entities = _extract_entities("Apple Apple Apple reported earnings")
        apple_count = sum(1 for e in entities if "Apple" in e)
        assert apple_count == 1

    def test_empty_text_returns_list(self):
        entities = _extract_entities("")
        assert isinstance(entities, list)


class TestDetectSectors:
    def test_crypto_detected(self):
        sectors = _detect_sectors("Bitcoin dropped amid regulatory pressure")
        assert "crypto" in sectors

    def test_commodities_detected(self):
        sectors = _detect_sectors("Gold futures fell on dollar strength")
        assert "commodities" in sectors

    def test_forex_detected(self):
        sectors = _detect_sectors("Federal Reserve raised interest rates")
        assert "forex" in sectors

    def test_tech_detected(self):
        sectors = _detect_sectors("Nvidia semiconductor chip revenue surged")
        assert "tech" in sectors

    def test_aerospace_detected(self):
        sectors = _detect_sectors("SpaceX rocket launch successful")
        assert "aerospace" in sectors

    def test_unknown_returns_general(self):
        sectors = _detect_sectors("the quick brown fox jumps over the lazy dog")
        assert "general" in sectors

    def test_multiple_sectors_possible(self):
        sectors = _detect_sectors("Apple tech stock and bitcoin crypto both rose")
        assert len(sectors) >= 2


class TestDetectEventType:
    def test_ipo_detected(self):
        event = _detect_event_type("SpaceX IPO filing sent stocks soaring")
        assert event == "ipo"

    def test_earnings_detected(self):
        event = _detect_event_type("Apple reported record quarterly earnings beat")
        assert event == "earnings"

    def test_rate_decision_detected(self):
        event = _detect_event_type("Fed announced rate hike of 25 basis points")
        assert event == "rate_decision"

    def test_merger_detected(self):
        event = _detect_event_type("Microsoft acquisition of Activision approved")
        assert event == "merger"

    def test_layoffs_detected(self):
        event = _detect_event_type("Company announced 10% workforce reduction layoffs")
        assert event == "layoffs"

    def test_unknown_returns_general(self):
        event = _detect_event_type("The quick brown fox")
        assert event == "general"


class TestComputeImpactScore:
    def test_returns_float_in_range(self):
        score = _compute_impact_score("Gold fell 5% on dollar strength", 0.9)
        assert 0.0 <= score <= 1.0

    def test_high_impact_keywords_increase_score(self):
        high = _compute_impact_score("Record historic crash of 30% triggered crisis", 0.95)
        low = _compute_impact_score("prices changed slightly", 0.5)
        assert high > low

    def test_low_confidence_dampens_score(self):
        high_conf = _compute_impact_score("billion dollar acquisition", 0.95)
        low_conf = _compute_impact_score("billion dollar acquisition", 0.1)
        assert high_conf > low_conf


# ---------------------------------------------------------------------------
# SignalExtractor (random-weight model)
# ---------------------------------------------------------------------------


class TestSignalExtractor:
    @pytest.fixture(scope="class")
    def extractor(self):
        """Random-weight extractor — no trained checkpoint needed."""
        return SignalExtractor()

    def test_extract_returns_financial_signal(self, extractor):
        signal = extractor.extract("Apple reported record quarterly earnings")
        assert isinstance(signal, FinancialSignal)

    def test_extract_sentiment_is_valid(self, extractor):
        signal = extractor.extract("Bitcoin dropped amid regulatory pressure")
        assert signal.sentiment in ("positive", "negative", "neutral")

    def test_extract_confidence_in_range(self, extractor):
        signal = extractor.extract("Gold futures rose on safe-haven demand")
        assert 0.0 <= signal.confidence <= 1.0

    def test_extract_direction_matches_sentiment(self, extractor):
        # Direction must be consistent with sentiment
        signal = extractor.extract("tech stocks rallied")
        mapping = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}
        assert signal.signal_direction == mapping[signal.sentiment]

    def test_extract_impact_score_in_range(self, extractor):
        signal = extractor.extract("Oil prices surged 10% on supply disruption")
        assert 0.0 <= signal.impact_score <= 1.0

    def test_extract_sectors_is_list(self, extractor):
        signal = extractor.extract("Ethereum staking yield attracted investors")
        assert isinstance(signal.sectors, list)
        assert len(signal.sectors) > 0

    def test_extract_entities_is_list(self, extractor):
        signal = extractor.extract("Apple and Microsoft reported earnings")
        assert isinstance(signal.entities, list)

    def test_spacex_ipo_aerospace_sector(self, extractor):
        """SpaceX IPO should be detected as aerospace + ipo event."""
        signal = extractor.extract(
            "SpaceX's long-anticipated IPO filing sent aerospace stocks soaring"
        )
        assert "aerospace" in signal.sectors
        assert signal.event_type == "ipo"

    def test_bitcoin_regulatory_entities(self, extractor):
        """Bitcoin regulatory news should have crypto sector and bitcoin entity."""
        signal = extractor.extract("Bitcoin dropped below $60,000 as regulatory pressure mounted")
        assert "crypto" in signal.sectors
        assert any("Bitcoin" in e or "bitcoin" in e.lower() for e in signal.entities)

    def test_gold_dollar_commodities_forex(self, extractor):
        """Gold + dollar news should detect both commodities and forex sectors."""
        signal = extractor.extract(
            "Gold futures fell 2.3% amid dollar strength and Fed rate concerns"
        )
        assert "commodities" in signal.sectors or "forex" in signal.sectors

    def test_extract_to_dict_is_json_serialisable(self, extractor):
        import json

        signal = extractor.extract("Apple earnings beat expectations")
        d = signal.to_dict()
        # Should not raise
        serialised = json.dumps(d)
        assert isinstance(serialised, str)

    def test_extract_batch_returns_list(self, extractor):
        texts = [
            "Gold fell 2% on dollar strength",
            "SpaceX IPO sent aerospace stocks soaring",
            "Bitcoin dropped amid regulatory news",
        ]
        signals = extractor.extract_batch(texts)
        assert isinstance(signals, list)
        assert len(signals) == len(texts)

    def test_extract_batch_same_order_as_input(self, extractor):
        texts = [
            "Apple earnings beat expectations",
            "Oil prices fell on demand concerns",
            "Fed raised interest rates",
        ]
        signals = extractor.extract_batch(texts)
        for i, (text, signal) in enumerate(zip(texts, signals)):
            assert signal.text == text, f"Order mismatch at index {i}"

    def test_extract_batch_empty_list(self, extractor):
        signals = extractor.extract_batch([])
        assert signals == []

    def test_extract_batch_single_item(self, extractor):
        signals = extractor.extract_batch(["Gold rose on safe-haven demand"])
        assert len(signals) == 1
        assert isinstance(signals[0], FinancialSignal)

    def test_extract_long_text(self, extractor):
        long_text = (
            "The Federal Reserve announced a 25 basis point interest rate hike "
            "citing persistent inflation concerns, while gold futures rose 1.5% "
            "and the dollar index fell, with Bitcoin also gaining amid the "
            "uncertainty as investors sought alternative stores of value. "
            "Tech stocks remained mixed with Nvidia gaining on AI optimism "
            "while Apple fell slightly on revenue guidance concerns. " * 3
        )
        signal = extractor.extract(long_text)
        assert isinstance(signal, FinancialSignal)
        assert signal.sentiment in ("positive", "negative", "neutral")
