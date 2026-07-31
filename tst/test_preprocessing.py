"""Tests for src/data/preprocessing.py, in particular the labeling formula and
time-based splitting -- the two places where a look-ahead bug would be most
damaging and least visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import build_features, make_label, time_splits


def _close_series(n=40, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n))), index=idx)
    return close


def test_make_label_direction_horizon_1_uses_close_t_over_close_t_minus_1():
    """For horizon=1, make_label's telescoping-sum formula reduces to the
    return between close[t-1] and close[t] (i.e. the *same* return already
    exposed to the model as the 'log_ret' feature at row t). This is not a
    leakage bug: build_sequences() (see test_sequences.py) always excludes a
    sample's own row from its feature window, so the label at t is only ever
    predicted from rows < t. This test locks the formula so a future edit
    can't silently shift it by one day without a test failing.
    """
    close = _close_series()
    df = pd.DataFrame({"Close": close})
    y = make_label(df, label_type="direction", horizon=1)

    expected_return = np.log(close / close.shift(1))
    expected_direction = (expected_return > 0).astype(int)

    aligned = y.dropna()
    pd.testing.assert_series_equal(
        aligned, expected_direction.loc[aligned.index], check_names=False, check_dtype=False
    )


def test_make_label_return_horizon_h_matches_telescoped_log_return():
    """For horizon=h, future_lr[t] should equal log(close[t+h-1] / close[t-1])."""
    close = _close_series()
    df = pd.DataFrame({"Close": close})
    horizon = 3
    y = make_label(df, label_type="return", horizon=horizon)

    n = len(close)
    expected = pd.Series(index=close.index, dtype=float)
    for t in range(1, n - horizon + 1):
        expected.iloc[t] = np.log(close.iloc[t + horizon - 1] / close.iloc[t - 1])

    common = y.dropna().index.intersection(expected.dropna().index)
    assert len(common) > 0
    np.testing.assert_allclose(y.loc[common].values, expected.loc[common].values, rtol=1e-10)


def test_make_label_invalid_type_raises():
    df = pd.DataFrame({"Close": _close_series()})
    with pytest.raises(ValueError):
        make_label(df, label_type="not_a_real_type")


def test_build_features_no_look_ahead_first_row_is_nan():
    close = _close_series()
    df = pd.DataFrame({"Close": close})
    feats = build_features(df)
    # log_ret at t=0 has no t-1 -> must be NaN, never silently filled with 0.
    assert np.isnan(feats["log_ret"].iloc[0])


def test_time_splits_are_chronological_and_non_overlapping():
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    df = pd.DataFrame({"x": range(200)}, index=idx)
    train_idx, val_idx, test_idx = time_splits(df, val_start=None, test_start="2020-06-01")

    assert len(train_idx) + len(val_idx) + len(test_idx) == len(idx)
    if len(train_idx) and len(val_idx):
        assert train_idx.max() < val_idx.min()
    if len(val_idx) and len(test_idx):
        assert val_idx.max() < test_idx.min()
    assert (test_idx >= pd.Timestamp("2020-06-01")).all()


def test_time_splits_auto_validation_is_about_10_percent_of_train_val():
    idx = pd.date_range("2020-01-01", periods=1000, freq="D")
    df = pd.DataFrame({"x": range(1000)}, index=idx)
    train_idx, val_idx, test_idx = time_splits(df, val_start=None, test_start="2022-06-01")

    n_trainval = len(train_idx) + len(val_idx)
    ratio = len(val_idx) / n_trainval
    assert 0.08 <= ratio <= 0.12


def test_time_splits_requires_test_start():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"x": range(10)}, index=idx)
    with pytest.raises(ValueError):
        time_splits(df, val_start=None, test_start=None)
