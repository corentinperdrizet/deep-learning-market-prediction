# -----------------------------
# File: src/serving/model_registry.py
# -----------------------------
"""Pure business logic for model inference -- no FastAPI/Pydantic here on
purpose, so it can be unit-tested (and reused from a notebook or a batch
job) without spinning up a web server.

A checkpoint is self-describing: train_lstm()/train_transformer() persist a
`model_config` dict alongside the state_dict (see src/training/trainer.py),
so load_model() reconstructs the exact architecture that produced a given
.pt file instead of hardcoding hyperparameters here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import torch

from ..data.config import DataConfig
from ..data.dataset import build_feature_frame
from ..data.paths import artifact_prefix, scaler_filename
from ..data.preprocessing import FEATURE_COLUMNS_DEFAULT
from ..data.scaling import transform_with_scaler
from ..models.lstm import LSTMClassifier
from ..models.transformer import TransformerTimeSeriesClassifier
from ..training.utils import get_device

ModelKind = Literal["lstm", "transformer"]
_MODEL_KINDS: tuple[ModelKind, ...] = ("lstm", "transformer")


class ModelNotFoundError(FileNotFoundError):
    pass


@dataclass
class PredictionResult:
    ticker: str
    model_kind: str
    as_of: str
    probability_up: float
    signal: int
    threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


def _checkpoint_path(model_kind: ModelKind, ticker: str, artifacts_dir: Path) -> Path:
    return artifacts_dir / f"{artifact_prefix(model_kind, ticker)}_classifier.pt"


def list_available_models(artifacts_dir: str = "data/artifacts") -> list[dict]:
    """Scan artifacts_dir for trained checkpoints, returning [{model_kind, ticker}]."""
    d = Path(artifacts_dir)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*_classifier.pt")):
        stem = p.stem[: -len("_classifier")]
        for kind in _MODEL_KINDS:
            if stem == kind:
                out.append({"model_kind": kind, "ticker": "BTC-USD"})
                break
            prefix = f"{kind}_"
            if stem.startswith(prefix):
                out.append({"model_kind": kind, "ticker": stem[len(prefix) :]})
                break
    return out


def load_model(model_kind: ModelKind, ticker: str, artifacts_dir: str = "data/artifacts"):
    """Load a checkpoint and rebuild its architecture from model_config.

    Returns (model, model_config, device). Raises ModelNotFoundError if no
    checkpoint exists, or ValueError if the checkpoint predates the
    model_config metadata (retrain to fix).
    """
    if model_kind not in _MODEL_KINDS:
        raise ValueError(f"Unknown model_kind {model_kind!r}, expected one of {_MODEL_KINDS}")

    ckpt_path = _checkpoint_path(model_kind, ticker, Path(artifacts_dir))
    if not ckpt_path.exists():
        raise ModelNotFoundError(
            f"No {model_kind} checkpoint for ticker={ticker!r} at {ckpt_path}. "
            f"Train one first (e.g. python -m src.training.run_{model_kind} --ticker {ticker})."
        )

    device = get_device()
    state = torch.load(ckpt_path, map_location=device)
    model_config = state.get("model_config")
    if not model_config:
        raise ValueError(
            f"Checkpoint at {ckpt_path} has no model_config metadata (it predates that feature). "
            "Retrain to produce a self-describing checkpoint."
        )

    if model_kind == "lstm":
        model = LSTMClassifier(
            input_dim=model_config["input_dim"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            bidirectional=model_config["bidirectional"],
        )
    else:
        model = TransformerTimeSeriesClassifier(
            input_dim=model_config["input_dim"],
            d_model=model_config["d_model"],
            n_heads=model_config["n_heads"],
            n_layers=model_config["n_layers"],
            dim_feedforward=model_config["dim_feedforward"],
            dropout=model_config["dropout"],
            pooling=model_config["pooling"],
            max_len=model_config["max_len"],
        )

    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model, model_config, device


def _load_threshold(model_kind: ModelKind, artifacts_dir: str) -> float:
    path = Path(artifacts_dir) / "thresholds.json"
    if not path.exists():
        return 0.5
    payload = json.loads(path.read_text())
    entry = payload.get(model_kind)
    if not entry:
        return 0.5
    return float(entry.get("theta", 0.5))


def build_latest_window(
    ticker: str, seq_len: int, artifacts_dir: str = "data/artifacts", start: str = "2018-01-01"
) -> tuple[np.ndarray, str]:
    """Download the latest data, apply the ticker's saved scaler, and return
    the most recent seq_len-day window ready for model input, plus the
    as-of date of its last row.
    """
    cfg = DataConfig(ticker=ticker, start=start)
    data = build_feature_frame(cfg)
    features = [c for c in FEATURE_COLUMNS_DEFAULT if c in data.columns]
    if len(data) < seq_len:
        raise ValueError(
            f"Not enough history for {ticker} to build a {seq_len}-day window (have {len(data)} rows)."
        )

    window_df = data.iloc[-seq_len:][features]
    # Loaded directly from artifacts_dir (not via data.scaling.load_scaler(),
    # which always resolves the project's default data/artifacts location)
    # so a caller-supplied artifacts_dir is honored consistently for both
    # the checkpoint and the scaler.
    scaler_path = Path(artifacts_dir) / scaler_filename(ticker)
    if not scaler_path.exists():
        raise ModelNotFoundError(
            f"No scaler found for ticker={ticker!r} at {scaler_path}. Train a model for this "
            "ticker first so its scaler is persisted."
        )
    scaler = joblib.load(scaler_path)
    X = transform_with_scaler(scaler, window_df)
    X = X.reshape(1, seq_len, -1).astype(np.float32)
    return X, str(data.index[-1])


def predict_latest(
    model_kind: ModelKind,
    ticker: str,
    artifacts_dir: str = "data/artifacts",
    start: str = "2018-01-01",
) -> PredictionResult:
    """End-to-end: load model, fetch the latest window, run inference."""
    model, model_config, device = load_model(model_kind, ticker, artifacts_dir)
    X, as_of = build_latest_window(ticker, model_config["seq_len"], artifacts_dir, start)

    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
    proba = float(1.0 / (1.0 + np.exp(-logits.cpu().numpy()[0])))

    theta = _load_threshold(model_kind, artifacts_dir)
    return PredictionResult(
        ticker=ticker,
        model_kind=model_kind,
        as_of=as_of,
        probability_up=proba,
        signal=int(proba > theta),
        threshold=theta,
    )
