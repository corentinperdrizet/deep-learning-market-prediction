# -----------------------------
# File: src/data/loaders.py
# -----------------------------
from __future__ import annotations
import pandas as pd
from pathlib import Path
import yfinance as yf
from .paths import data_dir


def _raw_cache_path(ticker: str, interval: str) -> Path:
    safe = ticker.replace("/", "-")
    return data_dir() / "raw" / f"{safe}_{interval}.parquet"


def download_ohlcv(
    ticker: str = "BTC-USD",
    start: str = "2018-01-01",
    end: str | None = None,
    interval: str = "1d",
    cache: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Download OHLCV via yfinance and return a clean DataFrame indexed by UTC datetime.

    Columns: [Open, High, Low, Close, Adj Close, Volume]
    """
    cache_path = _raw_cache_path(ticker, interval)
    if cache and cache_path.exists() and not force:
        df = pd.read_parquet(cache_path)
    else:
        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        if df.empty:
            raise RuntimeError(f"No data returned for {ticker} {interval}.")
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index().dropna(how="all")
        if cache:
            df.to_parquet(cache_path)
    return df


def ensure_daily_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure daily frequency with forward-fill OHLCV-friendly rules where appropriate.
    We *do not* fill price gaps by interpolation—only preserve existing rows.
    """
    df = df.copy()
    df = df[~df.index.duplicated(keep="first")]
    # For crypto, calendar is 24/7, so reindexing to full daily range may introduce no gaps.
    # We keep the natural index; quality checks will report any missing spans.
    return df


