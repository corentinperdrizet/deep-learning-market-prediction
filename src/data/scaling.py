# -----------------------------
# File: src/data/scaling.py
# -----------------------------
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict
from .paths import data_dir


SCALER_NAME = "scaler.joblib"


def fit_scaler(X_train: pd.DataFrame, robust: bool = False):
    scaler = RobustScaler() if robust else StandardScaler()
    scaler.fit(X_train.values)
    return scaler


def transform_with_scaler(scaler, X: pd.DataFrame) -> np.ndarray:
    return scaler.transform(X.values)


def save_scaler(scaler, name: str = SCALER_NAME) -> Path:
    path = data_dir() / "artifacts" / name
    joblib.dump(scaler, path)
    return path


def load_scaler(name: str = SCALER_NAME):
    path = data_dir() / "artifacts" / name
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found at {path}")
    return joblib.load(path)


