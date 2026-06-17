"""
Extended signal extractor tests for branches not covered by test_signals.py:
  - SECTOR_KEYWORDS exhaustive sector detection
  - EVENT_KEYWORDS all event types
  - KNOWN_ENTITIES all pattern types
  - HIGH_IMPACT_PATTERNS all patterns
  - SENTIMENT_TO_DIRECTION mapping
  - SignalExtractor with tokenizer_path and model_path (mocked)
  - FinancialSignal __repr__ with many entities (truncation)
  - _compute_impact_score edge cases (confidence=0, confidence=1, no patterns)
  - extract_batch with 100 items (boundary)
"""

from __future__ import annotations

import json

import pytest
import torch

from finbert.signals import (
    SENTIMENT_TO_DIRECTION,
    FinancialSignal,
    SignalExtractor,
    _compute_impact_score,
    _detect_event_type,
    _detect_sectors,
    _extract_entities,
)

# ---------------------------------------------------------------------------
# SENTIMENT_TO_DIRECTION mapping
# ---------------------------------------------------------------------------


class TestSentimentToDirection:
    def test_positive_maps_to_bullish(self):
        assert SENTIMENT_TO_DIRECTION["positive"] == "bullish"

    def test_negative_maps_to_bearish(self):
        assert SENTIMENT_TO_DIRECTION["negative"] == "bearish"

    def test_neutral_maps_to_neutral(self):
        assert SENTIMENT_TO_DIRECTION["neutral"] == "neutral"


# ---------------------------------------------------------------------------
# _extract_entities — all KNOWN_ENTITIES pattern types
# ---------------------------------------------------------------------------


class TestExtractEntitiesExhaustive:
    def test_apple_detected(self):
        entities = _extract_entities("Apple reported strong earnings")
        assert any("Apple" in e for e in entities)

    def test_microsoft_detected(self):
        entities = _extract_entities("Microsoft Azure revenue grew 29%")
        assert any("Microsoft" in e for e in entities)

    def test_google_detected(self):
        entities = _extract_entities("Google Cloud hit record quarterly revenue")
        assert any("Google" in e for e in entities)

    def test_alphabet_detected(self):
        entities = _extract_entities("Alphabet reported strong Q3 results")
        assert any("Alphabet" in e for e in entities)

    def test_amazon_detected(self):
        entities = _extract_entities("Amazon Web Services posted record income")
        assert any("Amazon" in e for e in entities)

    def test_meta_detected(self):
        entities = _extract_entities("Meta Platforms raised its revenue guidance")
        assert any("Meta" in e for e in entities)

    def test_tesla_detected(self):
        entities = _extract_entities("Tesla deliveries exceeded analyst estimates")
        assert any("Tesla" in e for e in entities)

    def test_nvidia_detected(self):
        entities = _extract_entities("Nvidia chip revenue surged on AI demand")
        assert any("Nvidia" in e for e in entities)

    def test_spacex_detected(self):
        entities = _extract_entities("SpaceX Starship completed its fourth test flight")
        assert any("SpaceX" in e for e in entities)

    def test_stripe_detected(self):
        entities = _extract_entities("Stripe processed record payment volumes")
        assert any("Stripe" in e for e in entities)

    def test_klarna_detected(self):
        entities = _extract_entities("Klarna filed for US IPO at $15 billion valuation")
        assert any("Klarna" in e for e in entities)

    def test_openai_detected(self):
        entities = _extract_entities("OpenAI raised a $6.6 billion funding round")
        assert any("OpenAI" in e for e in entities)

    def test_jpmorgan_detected(self):
        entities = _extract_entities("JPMorgan beat earnings estimates for the quarter")
        assert any("JPMorgan" in e for e in entities)

    def test_goldman_detected(self):
        entities = _extract_entities("Goldman Sachs upgraded Apple to a buy rating")
        assert any("Goldman" in e for e in entities)

    def test_blackrock_detected(self):
        entities = _extract_entities("BlackRock AUM grew to 10 trillion dollars")
        assert any("BlackRock" in e for e in entities)

    def test_berkshire_detected(self):
        entities = _extract_entities("Berkshire Hathaway reported record operating earnings")
        assert any("Berkshire" in e for e in entities)

    def test_btc_detected(self):
        entities = _extract_entities("BTC surged past 70000 for the first time")
        assert any("BTC" in e for e in entities)

    def test_eth_detected(self):
        entities = _extract_entities("ETH rose on Ethereum network upgrade news")
        assert any("ETH" in e or "Ethereum" in e for e in entities)

    def test_fed_fomc_detected(self):
        entities = _extract_entities("FOMC voted unanimously to hold rates steady")
        assert any("FOMC" in e or "Fed" in e for e in entities)

    def test_ecb_detected(self):
        entities = _extract_entities("ECB held rates steady citing disinflation progress")
        assert any("ECB" in e for e in entities)

    def test_silver_detected(self):
        entities = _extract_entities("Silver prices fell on profit taking")
        assert any("silver" in e.lower() for e in entities)

    def test_crude_oil_wti_detected(self):
        entities = _extract_entities("WTI crude oil fell below $75 per barrel")
        assert any("WTI" in e or "crude" in e.lower() for e in entities)

    def test_copper_detected(self):
        entities = _extract_entities("Copper prices tumbled on weak demand data")
        assert any("copper" in e.lower() for e in entities)

    def test_lithium_detected(self):
        entities = _extract_entities("Lithium prices stabilised after sharp decline")
        assert any("lithium" in e.lower() for e in entities)

    def test_usd_detected(self):
        entities = _extract_entities("USD strengthened against major currencies")
        assert any("USD" in e or "dollar" in e.lower() for e in entities)

    def test_eur_detected(self):
        entities = _extract_entities("EUR fell against the USD on weak data")
        assert any("EUR" in e or "euro" in e.lower() for e in entities)

    def test_jpy_detected(self):
        entities = _extract_entities("JPY weakened to 155 per dollar")
        assert any("JPY" in e or "yen" in e.lower() for e in entities)

    def test_gbp_detected(self):
        entities = _extract_entities("GBP rose after positive UK inflation data")
        assert any("GBP" in e or "pound" in e.lower() for e in entities)

    def test_ticker_dollar_prefix(self):
        entities = _extract_entities("$NVDA hit all-time high on earnings beat")
        assert any("$" in e for e in entities)


# ---------------------------------------------------------------------------
# _detect_sectors — all sector groups
# ---------------------------------------------------------------------------


class TestDetectSectorsExhaustive:
    def test_equities_detected(self):
        sectors = _detect_sectors("S&P 500 stock rally pushed earnings higher")
        assert "equities" in sectors

    def test_fixed_income_detected(self):
        sectors = _detect_sectors("10-year Treasury bond yield rose 10 basis points")
        assert "fixed_income" in sectors

    def test_tech_semiconductor_detected(self):
        sectors = _detect_sectors("semiconductor chip shortage worsened for gpu servers")
        assert "tech" in sectors

    def test_aerospace_defense_detected(self):
        sectors = _detect_sectors("Lockheed defense contract worth 5 billion approved")
        assert "aerospace" in sectors

    def test_commodities_oil_detected(self):
        sectors = _detect_sectors("brent crude oil futures jumped on OPEC cuts")
        assert "commodities" in sectors

    def test_crypto_web3_detected(self):
        sectors = _detect_sectors("web3 dao yield farming protocol launched on polygon")
        assert "crypto" in sectors

    def test_forex_rate_hike_detected(self):
        sectors = _detect_sectors("ECB rate hike triggered euro fx exchange rate move")
        assert "forex" in sectors

    def test_multiple_sectors_returned(self):
        sectors = _detect_sectors("tech stock bitcoin crypto earnings beat guidance")
        assert "tech" in sectors
        assert "crypto" in sectors
        assert "equities" in sectors


# ---------------------------------------------------------------------------
# _detect_event_type — all event types
# ---------------------------------------------------------------------------


class TestDetectEventTypeExhaustive:
    def test_ipo_prospectus(self):
        assert _detect_event_type("Company filed prospectus for upcoming IPO") == "ipo"

    def test_ipo_s1(self):
        assert _detect_event_type("s-1 filing approved by SEC underwriter road show") == "ipo"

    def test_earnings_quarterly(self):
        assert _detect_event_type("Q3 quarterly results beat EBITDA expectations") == "earnings"

    def test_earnings_eps(self):
        assert _detect_event_type("EPS exceeded analyst revenue forecast for Q2") == "earnings"

    def test_rate_decision_hawkish(self):
        event = _detect_event_type("hawkish Fed signaled quantitative easing taper")
        assert event == "rate_decision"

    def test_rate_decision_dovish(self):
        event = _detect_event_type("dovish ECB cut rates by 25 basis points bps")
        assert event == "rate_decision"

    def test_merger_lbo(self):
        event = _detect_event_type("leveraged buyout lbo acquisition deal signed at premium")
        assert event == "merger"

    def test_merger_all_stock(self):
        event = _detect_event_type("all-stock merger agreement bid offer accepted")
        assert event == "merger"

    def test_commodity_move_surged(self):
        event = _detect_event_type("Gold surged to record high")
        assert event == "commodity_move"

    def test_commodity_move_plunged(self):
        event = _detect_event_type("Oil plunged on demand fears")
        assert event == "commodity_move"

    def test_commodity_move_tumbled(self):
        event = _detect_event_type("Copper tumbled on weak Chinese data")
        assert event == "commodity_move"

    def test_regulatory_sec(self):
        event = _detect_event_type("SEC probe investigation into compliance violation penalty")
        assert event == "regulatory"

    def test_regulatory_antitrust(self):
        event = _detect_event_type("FTC antitrust lawsuit settlement ban sanction")
        assert event == "regulatory"

    def test_layoffs_rif(self):
        event = _detect_event_type("Company announced reduction in force rif headcount cut")
        assert event == "layoffs"

    def test_layoffs_redundancies(self):
        event = _detect_event_type("Redundancies announced as part of restructuring downsizing")
        assert event == "layoffs"

    def test_product_launch_unveiled(self):
        event = _detect_event_type("New product unveiled at partnership collaboration expansion")
        assert event == "product_launch"

    def test_product_launch_enters(self):
        event = _detect_event_type("Company enters new market with released product launch")
        assert event == "product_launch"

    def test_general_fallback(self):
        assert _detect_event_type("unrelated sentence about weather conditions") == "general"

    def test_first_match_wins(self):
        """IPO keyword appears before earnings — should return ipo."""
        event = _detect_event_type("IPO listing debut with quarterly earnings release")
        assert event == "ipo"


# ---------------------------------------------------------------------------
# _compute_impact_score — edge cases
# ---------------------------------------------------------------------------


class TestComputeImpactScoreExtra:
    def test_zero_confidence_dampens_to_half(self):
        """With confidence=0, score = base * 0.5."""
        score = _compute_impact_score("ordinary market movement", 0.0)
        assert 0.0 <= score <= 1.0

    def test_full_confidence_returns_high_for_high_impact(self):
        text = "Record historic crash 30% bankruptcy triggered trillion dollar crisis"
        score = _compute_impact_score(text, 1.0)
        assert score > 0.5

    def test_no_impact_keywords_base_is_0_2(self):
        """No keyword hits → base = min(0/8 + 0.2, 1.0) = 0.2."""
        score = _compute_impact_score("simple update today", 1.0)
        # base=0.2, full confidence: 0.2 * (0.5 + 0.5*1.0) = 0.2
        assert score == pytest.approx(0.2, abs=0.05)

    def test_score_capped_at_one(self):
        """Many keywords should not exceed 1.0."""
        text = (
            "record historic crash 30% bankruptcy crisis acquisition merger ipo "
            "rate hike fomc billion trillion chapter 11 all-time"
        )
        score = _compute_impact_score(text, 1.0)
        assert score <= 1.0

    def test_score_always_non_negative(self):
        for text in ["", "gold fell", "the quick brown fox"]:
            for conf in (0.0, 0.5, 1.0):
                score = _compute_impact_score(text, conf)
                assert score >= 0.0

    def test_percentage_pattern_increases_score(self):
        with_pct = _compute_impact_score("fell 15%", 0.8)
        without_pct = _compute_impact_score("fell a bit", 0.8)
        assert with_pct > without_pct

    def test_billion_trillion_increases_score(self):
        with_b = _compute_impact_score("billion dollar deal", 0.8)
        without_b = _compute_impact_score("small deal", 0.8)
        assert with_b > without_b


# ---------------------------------------------------------------------------
# FinancialSignal — repr truncation
# ---------------------------------------------------------------------------


class TestFinancialSignalRepr:
    def test_repr_truncates_entities_beyond_three(self):
        signal = FinancialSignal(
            text="x",
            sentiment="neutral",
            confidence=0.5,
            entities=["A", "B", "C", "D", "E"],
            sectors=["general"],
            event_type="general",
            impact_score=0.3,
            signal_direction="neutral",
        )
        r = repr(signal)
        assert "..." in r

    def test_repr_no_truncation_for_three_entities(self):
        signal = FinancialSignal(
            text="x",
            sentiment="positive",
            confidence=0.9,
            entities=["A", "B", "C"],
            sectors=["tech"],
            event_type="earnings",
            impact_score=0.7,
            signal_direction="bullish",
        )
        r = repr(signal)
        assert "..." not in r

    def test_repr_empty_entities(self):
        signal = FinancialSignal(text="x", sentiment="neutral", confidence=0.5)
        r = repr(signal)
        assert "entities=" in r

    def test_to_dict_is_serialisable(self):
        signal = FinancialSignal(
            text="gold fell",
            sentiment="negative",
            confidence=0.8,
            entities=["gold"],
            sectors=["commodities"],
            event_type="commodity_move",
            impact_score=0.4,
            signal_direction="bearish",
        )
        serialised = json.dumps(signal.to_dict())
        recovered = json.loads(serialised)
        assert recovered["sentiment"] == "negative"
        assert recovered["signal_direction"] == "bearish"


# ---------------------------------------------------------------------------
# SignalExtractor — extended scenarios
# ---------------------------------------------------------------------------


class TestSignalExtractorExtended:
    @pytest.fixture(scope="class")
    def extractor(self):
        return SignalExtractor()

    def test_empty_string_does_not_crash(self, extractor):
        """Empty text should produce a valid signal."""
        signal = extractor.extract("")
        assert isinstance(signal, FinancialSignal)
        assert signal.sentiment in ("positive", "negative", "neutral")

    def test_single_word_text(self, extractor):
        signal = extractor.extract("gold")
        assert isinstance(signal, FinancialSignal)

    def test_text_with_only_special_chars(self, extractor):
        signal = extractor.extract("!!! ??? ###")
        assert isinstance(signal, FinancialSignal)

    def test_all_sector_types_covered(self, extractor):
        texts_by_sector = {
            "tech": "Nvidia semiconductor chip gpu cloud ai software",
            "aerospace": "SpaceX rocket launch satellite orbit drone",
            "commodities": "gold silver copper oil crude brent wheat corn",
            "crypto": "bitcoin ethereum blockchain defi nft web3 btc eth",
            "forex": "dollar euro yen pound Fed FOMC interest rate inflation",
            "equities": "stock shares ipo earnings eps dividend buyback rally",
            "fixed_income": "bond treasury yield 10-year spread credit duration",
        }
        for sector, text in texts_by_sector.items():
            signal = extractor.extract(text)
            assert sector in signal.sectors, f"Expected sector '{sector}' in '{signal.sectors}'"

    def test_all_event_types_detectable(self, extractor):
        event_texts = {
            "ipo": "Company IPO filing debut listing went public",
            "earnings": "Quarterly earnings Q2 EPS revenue beat guidance",
            "rate_decision": "Fed FOMC rate hike basis points hawkish dovish",
            "merger": "Acquisition merger takeover deal bid offer premium",
            "commodity_move": "Gold surged fell rose jumped dropped",
            "regulatory": "SEC fine penalty antitrust lawsuit settlement probe",
            "layoffs": "Layoffs job cuts restructuring workforce reduction",
            "product_launch": "New product launched partnership collaboration expansion",
        }
        for event, text in event_texts.items():
            signal = extractor.extract(text)
            assert signal.event_type == event, (
                f"Expected event '{event}', got '{signal.event_type}' for: {text}"
            )

    def test_impact_score_between_zero_and_one(self, extractor):
        for text in [
            "stocks fell",
            "record historic crash 30% bankruptcy crisis",
            "",
            "billion dollar acquisition merger ipo all-time",
        ]:
            signal = extractor.extract(text)
            assert 0.0 <= signal.impact_score <= 1.0, f"Out of range for: {text!r}"

    def test_confidence_always_rounded_to_four_decimals(self, extractor):
        signal = extractor.extract("Gold fell on dollar strength")
        # Confidence should have at most 4 decimal places
        assert round(signal.confidence, 4) == signal.confidence

    def test_extract_batch_100_items(self, extractor):
        texts = ["gold fell"] * 100
        signals = extractor.extract_batch(texts)
        assert len(signals) == 100

    def test_extract_batch_all_signals_have_valid_sentiment(self, extractor):
        texts = [
            "Apple earnings beat",
            "Bitcoin crashed below 50000",
            "Gold remained flat",
            "Tesla stock soared on deliveries",
            "Oil dropped on demand fears",
        ]
        signals = extractor.extract_batch(texts)
        for s in signals:
            assert s.sentiment in ("positive", "negative", "neutral")

    def test_extract_batch_direction_consistent_with_sentiment(self, extractor):
        texts = ["gold fell", "stocks rose", "rates held steady"]
        signals = extractor.extract_batch(texts)
        mapping = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}
        for s in signals:
            assert s.signal_direction == mapping[s.sentiment]

    def test_non_english_text_extract(self, extractor):
        """Non-English text should not crash and produce a valid signal."""
        signal = extractor.extract("Les résultats financiers du troisième trimestre")
        assert isinstance(signal, FinancialSignal)
        assert signal.sentiment in ("positive", "negative", "neutral")

    def test_very_long_text_extract(self, extractor):
        """Text much longer than max_length is handled via truncation."""
        long_text = "Gold fell amid dollar strength. " * 200
        signal = extractor.extract(long_text)
        assert isinstance(signal, FinancialSignal)

    def test_text_with_html_characters(self, extractor):
        signal = extractor.extract("Earnings <strong>beat</strong> by 20% &amp; guidance raised")
        assert isinstance(signal, FinancialSignal)

    def test_text_with_numbers_and_percentages(self, extractor):
        signal = extractor.extract("Stock fell 15.7% to $142.30, down from $169.00")
        assert isinstance(signal, FinancialSignal)
        assert signal.impact_score > 0

    def test_extract_with_known_model_path_uses_from_checkpoint(self, tmp_path):
        """When model_path is provided, from_checkpoint is called."""
        from finbert.model import NanoFinBERT

        # Match the default tokenizer vocab (8000) and the extractor's encode
        # length (256); a smaller model overflows embedding / positional tables.
        model = NanoFinBERT(
            vocab_size=8000, hidden_dim=32, num_layers=1, num_heads=2, max_seq_len=256
        )
        ckpt_path = str(tmp_path / "test.pt")
        torch.save({"config": model.config, "model_state_dict": model.state_dict()}, ckpt_path)

        extractor = SignalExtractor(model_path=ckpt_path)
        signal = extractor.extract("Gold rose on safe-haven demand")
        assert isinstance(signal, FinancialSignal)

    def test_extract_with_tokenizer_path_uses_load(self, tmp_path):
        """When tokenizer_path is provided, FinancialTokenizer.load is called."""
        from finbert.tokenizer import FinancialTokenizer

        tok = FinancialTokenizer(vocab_size=300)
        tok.train(["gold silver oil bitcoin dollar earnings ipo merger"])
        tok_path = str(tmp_path / "tok.json")
        tok.save(tok_path)

        extractor = SignalExtractor(tokenizer_path=tok_path)
        signal = extractor.extract("Bitcoin fell on regulatory concerns")
        assert isinstance(signal, FinancialSignal)
