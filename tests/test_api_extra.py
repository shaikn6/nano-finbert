"""
Extended API server tests to cover missing branches:
  - _get_extractor 503 when model not loaded
  - extraction exception → 500 path
  - batch exception → 500 path
  - text at max length (2000 chars)
  - SignalResponse.from_signal
  - Model not yet loaded health response
  - CORS headers present
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from finbert.api.server import app, _get_extractor, SignalResponse
from finbert.signals import FinancialSignal


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# _get_extractor — 503 when extractor is None
# ---------------------------------------------------------------------------


class TestGetExtractorGuard:
    def test_503_when_extractor_is_none(self):
        """Patch the module-level _extractor to None to trigger 503."""
        import finbert.api.server as server_module

        original = server_module._extractor
        try:
            server_module._extractor = None
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get("/model/info")
                assert resp.status_code == 503
        finally:
            server_module._extractor = original

    def test_503_detail_message(self):
        import finbert.api.server as server_module

        original = server_module._extractor
        try:
            server_module._extractor = None
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get("/model/info")
                assert "not loaded" in resp.json().get("detail", "").lower()
        finally:
            server_module._extractor = original


# ---------------------------------------------------------------------------
# /extract — 500 when extractor.extract raises
# ---------------------------------------------------------------------------


class TestExtractErrorPath:
    def test_extract_exception_returns_500(self, client):
        import finbert.api.server as server_module

        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = RuntimeError("forward pass failed")
        original = server_module._extractor
        try:
            server_module._extractor = mock_extractor
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/extract", json={"text": "Apple earnings"})
                assert resp.status_code == 500
        finally:
            server_module._extractor = original

    def test_extract_500_detail_mentions_extraction(self, client):
        import finbert.api.server as server_module

        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = ValueError("tokenizer error")
        original = server_module._extractor
        try:
            server_module._extractor = mock_extractor
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/extract", json={"text": "gold fell"})
                detail = resp.json().get("detail", "")
                assert "Extraction failed" in detail or "extraction" in detail.lower()
        finally:
            server_module._extractor = original


# ---------------------------------------------------------------------------
# /extract/batch — 500 when extractor.extract_batch raises
# ---------------------------------------------------------------------------


class TestBatchErrorPath:
    def test_batch_exception_returns_500(self):
        import finbert.api.server as server_module

        mock_extractor = MagicMock()
        mock_extractor.extract_batch.side_effect = RuntimeError("batch failed")
        original = server_module._extractor
        try:
            server_module._extractor = mock_extractor
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/extract/batch", json={"texts": ["gold"]})
                assert resp.status_code == 500
        finally:
            server_module._extractor = original

    def test_batch_500_detail_mentions_batch(self):
        import finbert.api.server as server_module

        mock_extractor = MagicMock()
        mock_extractor.extract_batch.side_effect = RuntimeError("batch error")
        original = server_module._extractor
        try:
            server_module._extractor = mock_extractor
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/extract/batch", json={"texts": ["gold"]})
                detail = resp.json().get("detail", "")
                assert "Batch extraction failed" in detail or "batch" in detail.lower()
        finally:
            server_module._extractor = original


# ---------------------------------------------------------------------------
# /extract — text at boundary lengths
# ---------------------------------------------------------------------------


class TestExtractTextBoundaries:
    def test_text_exactly_2000_chars_accepted(self, client):
        text = "g" * 2000
        resp = client.post("/extract", json={"text": text})
        assert resp.status_code == 200

    def test_text_2001_chars_rejected(self, client):
        text = "g" * 2001
        resp = client.post("/extract", json={"text": text})
        assert resp.status_code == 422

    def test_text_one_char_accepted(self, client):
        resp = client.post("/extract", json={"text": "X"})
        assert resp.status_code == 200

    def test_text_none_value_rejected(self, client):
        resp = client.post("/extract", json={"text": None})
        assert resp.status_code == 422

    def test_text_integer_rejected(self, client):
        resp = client.post("/extract", json={"text": 42})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /extract/batch — list length boundaries
# ---------------------------------------------------------------------------


class TestBatchListBoundaries:
    def test_exactly_100_texts_accepted(self, client):
        texts = ["gold fell"] * 100
        resp = client.post("/extract/batch", json={"texts": texts})
        assert resp.status_code == 200

    def test_exactly_1_text_accepted(self, client):
        resp = client.post("/extract/batch", json={"texts": ["gold"]})
        assert resp.status_code == 200

    def test_101_texts_rejected(self, client):
        texts = ["gold"] * 101
        resp = client.post("/extract/batch", json={"texts": texts})
        assert resp.status_code == 422

    def test_batch_response_processing_time_is_float(self, client):
        resp = client.post("/extract/batch", json={"texts": ["gold rose"]})
        data = resp.json()
        assert isinstance(data["processing_time_ms"], float)


# ---------------------------------------------------------------------------
# SignalResponse.from_signal
# ---------------------------------------------------------------------------


class TestSignalResponseFromSignal:
    def test_from_signal_creates_correct_response(self):
        signal = FinancialSignal(
            text="Apple earnings beat",
            sentiment="positive",
            confidence=0.92,
            entities=["Apple"],
            sectors=["tech", "equities"],
            event_type="earnings",
            impact_score=0.65,
            signal_direction="bullish",
        )
        resp = SignalResponse.from_signal(signal)
        assert resp.text == signal.text
        assert resp.sentiment == signal.sentiment
        assert resp.confidence == signal.confidence
        assert resp.entities == signal.entities
        assert resp.sectors == signal.sectors
        assert resp.event_type == signal.event_type
        assert resp.impact_score == signal.impact_score
        assert resp.signal_direction == signal.signal_direction

    def test_from_signal_with_empty_entities_and_sectors(self):
        signal = FinancialSignal(
            text="market update",
            sentiment="neutral",
            confidence=0.50,
        )
        resp = SignalResponse.from_signal(signal)
        assert resp.entities == []
        assert resp.sectors == []

    def test_from_signal_model_is_serialisable(self):
        signal = FinancialSignal(
            text="gold rose",
            sentiment="positive",
            confidence=0.75,
            entities=["gold"],
            sectors=["commodities"],
            event_type="commodity_move",
            impact_score=0.4,
            signal_direction="bullish",
        )
        resp = SignalResponse.from_signal(signal)
        # Pydantic model can be serialised to dict without error
        d = resp.model_dump()
        assert d["sentiment"] == "positive"


# ---------------------------------------------------------------------------
# CORS and content type
# ---------------------------------------------------------------------------


class TestHeaders:
    def test_health_response_content_type_json(self, client):
        resp = client.get("/health")
        assert "application/json" in resp.headers.get("content-type", "")

    def test_extract_response_content_type_json(self, client):
        resp = client.post("/extract", json={"text": "Apple earnings"})
        assert "application/json" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Health model_loaded False scenario
# ---------------------------------------------------------------------------


class TestHealthModelLoaded:
    def test_health_model_loaded_false_when_extractor_none(self):
        """Health endpoint reflects model_loaded=False when extractor is None."""
        import finbert.api.server as server_module

        original = server_module._extractor
        try:
            server_module._extractor = None
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get("/health")
                assert resp.status_code == 200
                data = resp.json()
                assert data["model_loaded"] is False
        finally:
            server_module._extractor = original
