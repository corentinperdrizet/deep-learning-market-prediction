# -----------------------------
# File: src/serving/api.py
# -----------------------------
"""FastAPI inference service.

Run with:
    uvicorn src.serving.api:app --reload
or:
    make api

Routes are thin wrappers around src.serving.model_registry -- all the real
logic (loading a checkpoint, building the latest window, running inference)
lives there and is unit-tested independently of FastAPI.
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from ..training.utils import available_device_name
from . import model_registry
from .schemas import HealthResponse, ModelInfo, PredictionResponse

app = FastAPI(
    title="Deep Learning Market Prediction API",
    description="Serves next-period direction predictions from trained LSTM/Transformer checkpoints.",
    version="0.2.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", device=available_device_name())


@app.get("/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    return [ModelInfo(**m) for m in model_registry.list_available_models()]


@app.get("/predict/{ticker}", response_model=PredictionResponse)
def predict(
    ticker: str,
    model: Literal["lstm", "transformer"] = Query(default="lstm", description="Which trained model to use"),
) -> PredictionResponse:
    try:
        result = model_registry.predict_latest(model_kind=model, ticker=ticker)
    except model_registry.ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionResponse(**result.to_dict())
