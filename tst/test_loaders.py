"""Regression tests for src/data/loaders.py.

Covers a real bug found in review: the on-disk cache key used to be
(ticker, interval) only, so requesting a *different* --start/--end silently
returned data downloaded for a previous, mismatched date range instead of
triggering a fresh download. The cache key now includes (start, end).

All network access is mocked -- these tests never touch yfinance or the
network, and never write into the project's real data/ directory.
"""

from __future__ import annotations

import pandas as pd
import pytest

import src.data.loaders as loaders_mod


def _fake_ohlcv(n=5, start="2021-01-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": range(n),
            "High": range(n),
            "Low": range(n),
            "Close": range(n),
            "Adj Close": range(n),
            "Volume": range(n),
        },
        index=idx,
    ).astype(float)


@pytest.fixture
def patched_env(monkeypatch, tmp_path):
    """Redirect the cache to a tmp dir and count calls to yf.download."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    monkeypatch.setattr(loaders_mod, "data_dir", lambda: tmp_path)

    calls = {"n": 0}

    def fake_download(ticker, start, end, interval, auto_adjust, group_by, progress):
        calls["n"] += 1
        return _fake_ohlcv(start=start or "2021-01-01")

    monkeypatch.setattr(loaders_mod.yf, "download", fake_download)
    return calls


def test_same_date_range_hits_cache_only_downloads_once(patched_env):
    loaders_mod.download_ohlcv("BTC-USD", start="2021-01-01", end=None, interval="1d")
    loaders_mod.download_ohlcv("BTC-USD", start="2021-01-01", end=None, interval="1d")
    assert patched_env["n"] == 1


def test_different_start_triggers_a_fresh_download(patched_env):
    loaders_mod.download_ohlcv("BTC-USD", start="2021-01-01", end=None, interval="1d")
    loaders_mod.download_ohlcv("BTC-USD", start="2022-01-01", end=None, interval="1d")
    assert patched_env["n"] == 2


def test_different_end_triggers_a_fresh_download(patched_env):
    loaders_mod.download_ohlcv("BTC-USD", start="2021-01-01", end=None, interval="1d")
    loaders_mod.download_ohlcv("BTC-USD", start="2021-01-01", end="2021-06-01", interval="1d")
    assert patched_env["n"] == 2


def test_force_bypasses_cache(patched_env):
    loaders_mod.download_ohlcv("BTC-USD", start="2021-01-01", end=None, interval="1d")
    loaders_mod.download_ohlcv("BTC-USD", start="2021-01-01", end=None, interval="1d", force=True)
    assert patched_env["n"] == 2


def test_load_prices_reuses_download_ohlcv_cache(patched_env):
    s1 = loaders_mod.load_prices("BTC-USD", interval="1d", start="2021-01-01")
    s2 = loaders_mod.load_prices("BTC-USD", interval="1d", start="2021-01-01")
    assert patched_env["n"] == 1
    pd.testing.assert_series_equal(s1, s2, check_names=False, check_freq=False)


def test_ensure_daily_calendar_drops_duplicate_timestamps():
    idx = pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02"])
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx)
    out = loaders_mod.ensure_daily_calendar(df)
    assert len(out) == 2
    assert out["Close"].iloc[0] == 1.0  # keeps first occurrence
