"""
Tests for the FastAPI inference server.

Uses TestClient (synchronous) so no running server is needed.
The lifespan handler loads a random-weight model — no checkpoint required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from finbert.api.server import app


@pytest.fixture(scope="module")
def client():
    """Start the app (lifespan loads model) and return a test client."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health / info endpoints
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_model_loaded_true(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data


class TestModelInfo:
    def test_model_info_returns_200(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200

    def test_model_info_has_architecture(self, client):
        data = client.get("/model/info").json()
        assert data["architecture"] == "NanoFinBERT"

    def test_model_info_has_parameter_count(self, client):
        data = client.get("/model/info").json()
        assert data["total_parameters"] > 0

    def test_model_info_has_expected_keys(self, client):
        data = client.get("/model/info").json()
        for key in ("vocab_size", "hidden_dim", "num_layers", "num_heads", "max_seq_len"):
            assert key in data, f"Missing key: {key}"

    def test_model_info_device_is_cpu(self, client):
        data = client.get("/model/info").json()
        assert data["device"] == "cpu"


# ---------------------------------------------------------------------------
# /extract endpoint
# ---------------------------------------------------------------------------


class TestExtract:
    def test_extract_returns_200(self, client):
        resp = client.post("/extract", json={"text": "Apple reported record earnings"})
        assert resp.status_code == 200

    def test_extract_response_has_all_fields(self, client):
        resp = client.post("/extract", json={"text": "Gold fell on dollar strength"})
        data = resp.json()
        for field in (
            "text",
            "sentiment",
            "confidence",
            "entities",
            "sectors",
            "event_type",
            "impact_score",
            "signal_direction",
        ):
            assert field in data, f"Missing field: {field}"

    def test_extract_sentiment_is_valid(self, client):
        resp = client.post("/extract", json={"text": "Bitcoin surged to record highs"})
        data = resp.json()
        assert data["sentiment"] in ("positive", "negative", "neutral")

    def test_extract_confidence_in_range(self, client):
        resp = client.post("/extract", json={"text": "SpaceX IPO sent aerospace stocks soaring"})
        data = resp.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_extract_direction_is_valid(self, client):
        resp = client.post("/extract", json={"text": "Oil prices fell on demand concerns"})
        data = resp.json()
        assert data["signal_direction"] in ("bullish", "bearish", "neutral")

    def test_extract_entities_is_list(self, client):
        resp = client.post("/extract", json={"text": "Apple and Microsoft reported earnings"})
        data = resp.json()
        assert isinstance(data["entities"], list)

    def test_extract_sectors_is_list(self, client):
        resp = client.post("/extract", json={"text": "Ethereum blockchain defi protocol"})
        data = resp.json()
        assert isinstance(data["sectors"], list)

    def test_extract_text_echoed_back(self, client):
        text = "Tesla stock fell after disappointing deliveries"
        resp = client.post("/extract", json={"text": text})
        assert resp.json()["text"] == text

    def test_extract_empty_text_returns_422(self, client):
        resp = client.post("/extract", json={"text": ""})
        assert resp.status_code == 422

    def test_extract_missing_text_returns_422(self, client):
        resp = client.post("/extract", json={})
        assert resp.status_code == 422

    def test_extract_spacex_ipo_aerospace(self, client):
        resp = client.post(
            "/extract",
            json={"text": "SpaceX IPO filing sent aerospace stocks soaring to record highs"},
        )
        data = resp.json()
        assert "aerospace" in data["sectors"]
        assert data["event_type"] == "ipo"

    def test_extract_gold_dollar_commodities(self, client):
        resp = client.post(
            "/extract",
            json={"text": "Gold futures fell 2.3% amid dollar strength and Fed rate concerns"},
        )
        data = resp.json()
        assert "commodities" in data["sectors"] or "forex" in data["sectors"]


# ---------------------------------------------------------------------------
# /extract/batch endpoint
# ---------------------------------------------------------------------------


class TestExtractBatch:
    def test_batch_returns_200(self, client):
        resp = client.post(
            "/extract/batch",
            json={"texts": ["Gold rose", "Bitcoin fell", "Apple earnings beat"]},
        )
        assert resp.status_code == 200

    def test_batch_response_has_signals_key(self, client):
        resp = client.post("/extract/batch", json={"texts": ["Gold rose"]})
        data = resp.json()
        assert "signals" in data
        assert "count" in data
        assert "processing_time_ms" in data

    def test_batch_count_matches_input(self, client):
        texts = ["text one", "text two", "text three"]
        resp = client.post("/extract/batch", json={"texts": texts})
        data = resp.json()
        assert data["count"] == len(texts)
        assert len(data["signals"]) == len(texts)

    def test_batch_single_text(self, client):
        resp = client.post("/extract/batch", json={"texts": ["Gold rose on safe-haven demand"]})
        data = resp.json()
        assert data["count"] == 1

    def test_batch_processing_time_is_positive(self, client):
        resp = client.post("/extract/batch", json={"texts": ["Apple earnings beat"]})
        data = resp.json()
        assert data["processing_time_ms"] >= 0

    def test_batch_empty_list_returns_422(self, client):
        resp = client.post("/extract/batch", json={"texts": []})
        assert resp.status_code == 422

    def test_batch_each_signal_has_required_fields(self, client):
        resp = client.post(
            "/extract/batch",
            json={"texts": ["Gold rose", "Bitcoin fell"]},
        )
        for signal in resp.json()["signals"]:
            for field in ("text", "sentiment", "confidence", "signal_direction"):
                assert field in signal

    def test_batch_order_preserved(self, client):
        texts = [
            "Apple earnings beat expectations",
            "Oil prices fell on demand concerns",
            "Fed raised interest rates",
        ]
        resp = client.post("/extract/batch", json={"texts": texts})
        signals = resp.json()["signals"]
        for text, signal in zip(texts, signals):
            assert signal["text"] == text

    def test_batch_too_many_texts_returns_422(self, client):
        texts = ["gold"] * 101
        resp = client.post("/extract/batch", json={"texts": texts})
        assert resp.status_code == 422
