# -----------------------------
# File: src/serving/schemas.py
# -----------------------------
from __future__ import annotations

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    model_kind: str
    ticker: str


class PredictionResponse(BaseModel):
    ticker: str
    model_kind: str
    as_of: str = Field(description="Timestamp of the last bar used in the input window")
    probability_up: float = Field(ge=0.0, le=1.0, description="P(next-period return > 0)")
    signal: int = Field(description="1 = long, 0 = flat, at the model's saved decision threshold")
    threshold: float = Field(ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str
    device: str
