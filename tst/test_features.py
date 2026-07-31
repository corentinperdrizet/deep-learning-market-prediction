from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features import (
    calendar_features,
    log_returns,
    macd,
    rolling_returns,
    rolling_volatility,
    rsi,
)


def test_log_returns_matches_manual_formula():
    close = pd.Series([100.0, 110.0, 121.0, 108.9])
    lr = log_returns(close)
    assert np.isnan(lr.iloc[0])
    expected = np.log(close / close.shift(1))
    pd.testing.assert_series_equal(lr.iloc[1:], expected.iloc[1:], check_names=False)


def test_rolling_volatility_is_nan_before_window_then_positive():
    rng = np.random.default_rng(1)
    lr = pd.Series(rng.normal(0, 0.01, size=50))
    vol = rolling_volatility(lr, window=20)
    assert vol.iloc[:19].isna().all()
    assert (vol.iloc[19:].dropna() >= 0).all()


def test_rsi_bounded_between_0_and_100():
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=100))))
    r = rsi(close, window=14)
    valid = r.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_is_100_for_strictly_increasing_prices():
    close = pd.Series(np.arange(1, 30, dtype=float))
    r = rsi(close, window=14)
    # No losses at all -> RS -> inf -> RSI -> 100
    assert np.isclose(r.iloc[-1], 100.0, atol=1e-6)


def test_macd_hist_equals_macd_minus_signal():
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=100))))
    out = macd(close, fast=12, slow=26, signal=9)
    assert list(out.columns) == ["macd", "macd_signal", "macd_hist"]
    pd.testing.assert_series_equal(out["macd_hist"], out["macd"] - out["macd_signal"], check_names=False)


def test_rolling_returns_ret_1_matches_pct_change():
    close = pd.Series([100.0, 105.0, 103.0, 110.0])
    out = rolling_returns(close, [1, 3])
    pd.testing.assert_series_equal(out["ret_1"], close.pct_change(1), check_names=False)
    assert "ret_3" in out.columns


def test_calendar_features_sin_cos_unit_circle():
    idx = pd.date_range("2020-01-01", periods=14, freq="D")
    feats = calendar_features(idx)
    norm = feats["dow_sin"] ** 2 + feats["dow_cos"] ** 2
    assert np.allclose(norm, 1.0, atol=1e-10)
    assert feats["dow"].between(0, 6).all()
