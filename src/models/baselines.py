# src/models/baselines.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal

import numpy as np
import pandas as pd

# Sklearn baselines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier  # optional
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


ArrayLike = np.ndarray
Pooling = Literal["last", "mean", "flatten_last_k"]


def sequences_to_tabular(
    X_seq: ArrayLike,
    pooling: Pooling = "last",
    k: int = 5,
) -> ArrayLike:
    """
    Convert 3D sequences (N, T, F) to 2D tabular (N, F') for non-sequential models.

    - "last": use the last timestep features (N, F)
    - "mean": mean-pool across time (N, F)
    - "flatten_last_k": flatten last k timesteps -> (N, k*F)
    """
    if X_seq.ndim != 3:
        raise ValueError(f"Expected X_seq shape (N,T,F), got {X_seq.shape}")

    if pooling == "last":
        return X_seq[:, -1, :]
    elif pooling == "mean":
        return X_seq.mean(axis=1)
    elif pooling == "flatten_last_k":
        if k <= 0:
            raise ValueError("k must be > 0 for flatten_last_k")
        if X_seq.shape[1] < k:
            raise ValueError(f"X_seq has only {X_seq.shape[1]} timesteps, < k={k}")
        return X_seq[:, -k:, :].reshape(X_seq.shape[0], -1)
    else:
        raise ValueError(f"Unknown pooling={pooling}")


@dataclass
class BuyAndHoldClassifier:
    """Always predicts the base rate (train prevalence of class 1).

    This is a proper, trivial probabilistic baseline.
    """

    p_hat_: Optional[float] = None

    def fit(self, y_train: ArrayLike) -> "BuyAndHoldClassifier":
        y = np.asarray(y_train).astype(float)
        # Numerical stability for edge cases
        eps = 1e-6
        self.p_hat_ = float(np.clip(y.mean(), eps, 1 - eps))
        return self

    def predict_proba(self, n: int) -> ArrayLike:
        if self.p_hat_ is None:
            raise RuntimeError("Model not fitted. Call fit(y_train) first.")
        p1 = np.full(shape=(n,), fill_value=self.p_hat_, dtype=float)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


@dataclass
class LogisticRegressionTabular:
    """Logistic regression on tabularized sequence features.

    Parameters
    ----------
    pooling: how to reduce sequences to tabular ("last", "mean", "flatten_last_k")
    k: if pooling=="flatten_last_k", number of last timesteps to flatten
    C, penalty, max_iter, class_weight: passed to sklearn.LogisticRegression
    """

    pooling: Pooling = "last"
    k: int = 5
    C: float = 1.0
    penalty: str = "l2"
    max_iter: int = 200
    class_weight: Optional[str | dict] = None
    random_state: int = 42

    def _make_pipeline(self) -> Pipeline:
        lr = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            solver="lbfgs",
            random_state=self.random_state,
        )
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", lr),
        ])

    def fit(self, X_train: ArrayLike, y_train: ArrayLike) -> "LogisticRegressionTabular":
        X_tab = sequences_to_tabular(X_train, pooling=self.pooling, k=self.k)
        self.pipe_ = self._make_pipeline()
        self.pipe_.fit(X_tab, y_train)
        return self

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        X_tab = sequences_to_tabular(X, pooling=self.pooling, k=self.k)
        proba = self.pipe_.predict_proba(X_tab)
        return proba


@dataclass
class XGBTabular:
    """XGBoost classifier on flattened/pooled sequences (optional dependency)."""

    pooling: Pooling = "flatten_last_k"
    k: int = 5
    params: Optional[Dict[str, Any]] = None

    def fit(self, X_train: ArrayLike, y_train: ArrayLike) -> "XGBTabular":
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed. pip install xgboost")
        X_tab = sequences_to_tabular(X_train, pooling=self.pooling, k=self.k)
        default = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        )
        p = {**default, **(self.params or {})}
        self.model_ = XGBClassifier(**p)
        self.model_.fit(X_tab, y_train)
        return self

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        X_tab = sequences_to_tabular(X, pooling=self.pooling, k=self.k)
        p1 = self.model_.predict_proba(X_tab)[:, 1]
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


@dataclass
class SMACrossoverClassifier:
    """Rule-based SMA(lookback_short) vs SMA(lookback_long) signal.

    Produces class probabilities 0 or 1 from price crossovers.

    Important: you must pass a price series aligned with the *features index* of
    your labels. We assume daily bars and a 1-bar horizon for direction labels.
    The prediction at t uses info up to t (SMAs), targets compare P_{t+1}/P_t.
    """

    lookback_short: int = 50
    lookback_long: int = 200

    def fit(self, prices: pd.Series) -> "SMACrossoverClassifier":
        if not isinstance(prices, pd.Series):
            raise ValueError("prices must be a pandas Series with a DateTimeIndex")
        if self.lookback_short >= self.lookback_long:
            raise ValueError("lookback_short must be < lookback_long")
        self._prices_index_ = prices.index
        self.sma_short_ = prices.rolling(self.lookback_short).mean()
        self.sma_long_ = prices.rolling(self.lookback_long).mean()
        # Signal is 1 if SMA_short > SMA_long else 0; NaNs to 0 initially
        raw = (self.sma_short_ > self.sma_long_).astype(float).fillna(0.0)
        self.signal_ = raw
        return self

    def predict_proba(self, index_like: pd.Index) -> ArrayLike:
        if not hasattr(self, "signal_"):
            raise RuntimeError("Model not fitted. Call fit(prices) first.")
        # Align to provided index (e.g., the y index window). Use reindex with ffill
        sig = self.signal_.reindex(index_like).ffill().fillna(0.0)
        p1 = sig.to_numpy(dtype=float)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)
