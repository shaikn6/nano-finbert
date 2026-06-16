"""
FastAPI inference server for NanoFinBERT.

Endpoints:
    POST /extract          — single text → FinancialSignal JSON
    POST /extract/batch    — list of texts → list[FinancialSignal] JSON
    GET  /health           — liveness check
    GET  /model/info       — architecture metadata

Run with:
    uvicorn finbert.api.server:app --host 0.0.0.0 --port 8000 --reload

Or via docker-compose:
    docker-compose up
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from finbert.signals import FinancialSignal, SignalExtractor

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Financial text to analyse (headline, report excerpt, etc.)",
        examples=["SpaceX's long-anticipated IPO filing sent aerospace stocks soaring"],
    )


class BatchExtractRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of financial texts to analyse (max 100 per request)",
    )


class SignalResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    entities: list[str]
    sectors: list[str]
    event_type: str
    impact_score: float
    signal_direction: str

    @classmethod
    def from_signal(cls, signal: FinancialSignal) -> SignalResponse:
        return cls(**signal.to_dict())


class BatchSignalResponse(BaseModel):
    signals: list[SignalResponse]
    count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    architecture: str
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    max_seq_len: int
    total_parameters: int
    device: str


# ---------------------------------------------------------------------------
# App lifecycle — load model once at startup
# ---------------------------------------------------------------------------

_extractor: SignalExtractor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the SignalExtractor (and model) once at startup, reuse across requests."""
    global _extractor

    model_path = os.getenv("MODEL_PATH")  # optional trained checkpoint
    tokenizer_path = os.getenv("TOKENIZER_PATH")  # optional saved tokenizer
    device = os.getenv("DEVICE", "cpu")

    _extractor = SignalExtractor(
        model_path=model_path or None,
        tokenizer_path=tokenizer_path or None,
        device=device,
    )
    print(
        f"[nano-finbert] Model loaded | "
        f"params={_extractor.model.count_parameters():,} | "
        f"device={device}"
    )
    yield
    # Clean up on shutdown (nothing to do for CPU inference)
    _extractor = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="nano-finbert",
    description=(
        "Educational tiny transformer for financial signal extraction. "
        "Converts raw financial text into structured trading signals."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _get_extractor() -> SignalExtractor:
    if _extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again shortly.")
    return _extractor


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Liveness check — returns 200 when the service is ready."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        model_loaded=_extractor is not None,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["system"])
async def model_info() -> ModelInfoResponse:
    """Return architecture metadata for the loaded model."""
    extractor = _get_extractor()
    cfg = extractor.model.config
    return ModelInfoResponse(
        architecture="NanoFinBERT",
        vocab_size=cfg["vocab_size"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        max_seq_len=cfg["max_seq_len"],
        total_parameters=extractor.model.count_parameters(),
        device=extractor.device,
    )


@app.post("/extract", response_model=SignalResponse, tags=["inference"])
async def extract_signal(request: ExtractRequest) -> SignalResponse:
    """
    Extract a structured financial signal from a single text string.

    Returns sentiment, confidence, detected entities, sectors, event type,
    estimated market impact, and signal direction (bullish/bearish/neutral).
    """
    extractor = _get_extractor()
    try:
        signal = extractor.extract(request.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc
    return SignalResponse.from_signal(signal)


@app.post("/extract/batch", response_model=BatchSignalResponse, tags=["inference"])
async def extract_batch(request: BatchExtractRequest) -> BatchSignalResponse:
    """
    Extract structured financial signals from a batch of texts.

    Processes up to 100 texts in a single batched forward pass.
    Returns signals in the same order as the input list.
    """
    extractor = _get_extractor()
    if len(request.texts) > 100:
        raise HTTPException(
            status_code=422,
            detail="Maximum 100 texts per batch request.",
        )
    try:
        t0 = time.perf_counter()
        signals = extractor.extract_batch(request.texts)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch extraction failed: {exc}") from exc

    return BatchSignalResponse(
        signals=[SignalResponse.from_signal(s) for s in signals],
        count=len(signals),
        processing_time_ms=round(elapsed_ms, 2),
    )
