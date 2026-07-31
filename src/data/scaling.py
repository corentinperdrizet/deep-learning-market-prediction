# -----------------------------
# File: src/data/scaling.py
# -----------------------------
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from .paths import data_dir, scaler_filename


def fit_scaler(X_train: pd.DataFrame, robust: bool = False):
    scaler = RobustScaler() if robust else StandardScaler()
    scaler.fit(X_train.values)
    return scaler


def transform_with_scaler(scaler, X: pd.DataFrame) -> np.ndarray:
    # Handle empty splits gracefully
    if X.shape[0] == 0:
        return np.empty((0, X.shape[1]), dtype=float)
    return scaler.transform(X.values)


def save_scaler(scaler, ticker: str = "BTC-USD", name: str | None = None) -> Path:
    path = data_dir() / "artifacts" / (name or scaler_filename(ticker))
    joblib.dump(scaler, path)
    return path


def load_scaler(ticker: str = "BTC-USD", name: str | None = None):
    path = data_dir() / "artifacts" / (name or scaler_filename(ticker))
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found at {path}")
    return joblib.load(path)
