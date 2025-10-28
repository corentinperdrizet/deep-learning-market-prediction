# -----------------------------
# File: src/data/features.py
# -----------------------------
from __future__ import annotations
import numpy as np
import pandas as pd

# --- Technical indicators implemented locally (no external TA package) ---

def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def rolling_volatility(log_ret: pd.Series, window: int = 20) -> pd.Series:
    return log_ret.rolling(window, min_periods=window).std()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(alpha=1/window, adjust=False).mean()
    roll_down = pd.Series(down, index=close.index).ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def rolling_returns(close: pd.Series, windows: list[int]) -> pd.DataFrame:
    out = {}
    for w in windows:
        out[f"ret_{w}"] = close.pct_change(w)
    return pd.DataFrame(out)


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    # For daily crypto, we include day-of-week cyclic encodings
    dow = idx.dayofweek
    sin7 = np.sin(2 * np.pi * dow / 7)
    cos7 = np.cos(2 * np.pi * dow / 7)
    return pd.DataFrame({"dow": dow, "dow_sin": sin7, "dow_cos": cos7}, index=idx)


