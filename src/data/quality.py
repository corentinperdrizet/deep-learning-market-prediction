# -----------------------------
# File: src/data/quality.py
# -----------------------------
from __future__ import annotations
import pandas as pd


def check_missingness(df: pd.DataFrame) -> pd.Series:
    """Return per-column NA counts."""
    return df.isna().sum()


def check_duplicates(df: pd.DataFrame) -> int:
    """Return number of duplicate timestamps."""
    return int(df.index.duplicated().sum())


def detect_calendar_gaps(df: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
    """Detect missing timestamps relative to a given frequency.
    For crypto daily, gaps are rare; still useful for robustness.
    """
    rng = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz=df.index.tz)
    missing = rng.difference(df.index)
    return pd.DataFrame({"missing_ts": missing})


