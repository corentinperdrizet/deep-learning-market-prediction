# -----------------------------
# File: src/data/preprocessing.py
# -----------------------------
from __future__ import annotations
import pandas as pd
from typing import Tuple, Sequence
from .features import (
    log_returns, rolling_volatility, rsi, macd, rolling_returns, calendar_features
)

FEATURE_COLUMNS_DEFAULT = [
    # prices-derived
    "log_ret", "vol_20",
    # RSI
    "rsi_14",
    # MACD
    "macd", "macd_signal", "macd_hist",
    # multi-horizon returns
    "ret_1", "ret_3", "ret_7", "ret_14",
    # calendar
    "dow", "dow_sin", "dow_cos",
]

def build_features(
    df: pd.DataFrame,
    ret_windows: Sequence[int] = (1, 3, 7, 14),
    vol_window: int = 20,
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """
    Construct core features from OHLCV data.
    Automatically handles DataFrames with columns like ('Close', 'BTC-USD').
    """
    # Robust extraction of closing price (handles MultiIndex)
    if "Close" in df.columns:
        close = df["Close"]
    else:
        close = df.filter(like="Close").iloc[:, 0]

    feats = pd.DataFrame(index=df.index)
    feats["log_ret"] = log_returns(close)
    feats["vol_20"] = rolling_volatility(feats["log_ret"], window=vol_window)
    feats["rsi_14"] = rsi(close, window=rsi_window)
    feats = feats.join(macd(close, fast=macd_fast, slow=macd_slow, signal=macd_signal))
    feats = feats.join(rolling_returns(close, list(ret_windows)))
    feats = feats.join(calendar_features(df.index))
    return feats

def make_label(df: pd.DataFrame, label_type: str = "direction", horizon: int = 1) -> pd.Series:
    """
    Create label series aligned with features (no look-ahead).
    - direction: 1 if log return over horizon > 0 else 0
    - return: log return over horizon
    """
    lr = log_returns(df["Close"] if "Close" in df.columns else df.filter(like="Close").iloc[:, 0])
    future_lr = lr.shift(-horizon + 1).rolling(horizon).sum()

    if label_type == "direction":
        return (future_lr > 0).astype(int)
    elif label_type == "return":
        return future_lr
    else:
        raise ValueError("label_type must be 'direction' or 'return'")

def time_splits(
    df: pd.DataFrame,
    val_start: str | None,
    test_start: str | None
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    Split time index into train, validation, and test sets.
    If val_start is None, allocate the last 10% (>=1 sample) of data BEFORE test_start to validation.
    """
    idx = df.index
    if test_start is None:
        raise ValueError("test_start must be provided for clear OOS evaluation")

    # Everything strictly before test_start is train+val candidate
    mask_trainval = idx < test_start
    idx_trainval = idx[mask_trainval]

    if val_start is None:
        n_tv = len(idx_trainval)
        if n_tv == 0:
            # No train/val data at all; all in test
            return idx[:0], idx[:0], idx[idx >= test_start]
        n_val = max(1, int(round(0.10 * n_tv)))  # 10% for validation (at least 1)
        split_point = idx_trainval[-n_val]       # first index of validation segment
        val_start = split_point.isoformat()

    train_mask = idx < val_start
    val_mask = (idx >= val_start) & (idx < test_start)
    test_mask = idx >= test_start
    return idx[train_mask], idx[val_mask], idx[test_mask]
