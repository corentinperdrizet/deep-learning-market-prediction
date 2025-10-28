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


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """YFinance can return MultiIndex columns like (Field, Ticker).
    Standardize to single-index columns: ["Open","High","Low","Close","Adj Close","Volume"].
    """
    if isinstance(df.columns, pd.MultiIndex):
        # If single ticker, drop the 2nd level (ticker)
        if len(df.columns.levels[1]) == 1:
            df = df.droplevel(1, axis=1)
        else:
            # If multiple tickers in the future, keep Field_Ticker naming
            df.columns = [f"{c0}_{c1}" for c0, c1 in df.columns]
    return df


def download_ohlcv(
    ticker: str = "BTC-USD",
    start: str = "2018-01-01",
    end: str | None = None,
    interval: str = "1d",
    cache: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Download OHLCV via yfinance and return a clean DataFrame indexed by UTC datetime.

    Columns (single ticker): [Open, High, Low, Close, Adj Close, Volume]
    """
    cache_path = _raw_cache_path(ticker, interval)
    if cache and cache_path.exists() and not force:
        df = pd.read_parquet(cache_path)
    else:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            group_by="column",
            progress=False,
        )
        if df.empty:
            raise RuntimeError(f"No data returned for {ticker} {interval}.")
        df = _flatten_yf_columns(df)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index().dropna(how="all")
        # Ensure expected columns exist
        expected_cols = {"Open","High","Low","Close","Adj Close","Volume"}
        if expected_cols.issubset(set(df.columns)):
            df = df[["Open","High","Low","Close","Adj Close","Volume"]]
        if cache:
            df.to_parquet(cache_path)
    return df


def ensure_daily_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure daily frequency with forward-fill OHLCV-friendly rules where appropriate.
    We *do not* fill price gaps by interpolation—only preserve existing rows.
    """
    df = df.copy()
    df = df[~df.index.duplicated(keep="first")]
    return df

def load_prices(ticker: str, interval: str = "1d") -> pd.Series:
    df = yf.download(ticker, period="max", interval=interval, auto_adjust=True, progress=False)
    if "Close" not in df.columns:
        raise ValueError("Downloaded data missing 'Close' column")
    s = df["Close"].copy()
    s.name = ticker
    return s
