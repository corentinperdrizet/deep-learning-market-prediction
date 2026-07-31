from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import backtest


def _idx(n):
    return pd.date_range("2022-01-01", periods=n, freq="D")


def test_execution_lag_is_exactly_one_bar():
    """A desired signal set on day t must only affect the position (and thus
    PnL) starting on day t+1 -- this is the core anti-look-ahead guarantee.
    """
    n = 6
    idx = _idx(n)
    ret = pd.Series([0.0, 0.0, 0.10, 0.0, 0.0, 0.0], index=idx)  # a known return spike on day 2
    signal = pd.Series([0, 0, 1, 0, 0, 0], index=idx)  # decide to go long on day 2

    res = backtest(ret, signal, fees_bps=0.0, slippage_bps=0.0)
    df = res.df

    # Position on day 2 (the day the signal fires) must still be 0 ...
    assert df["pos"].iloc[2] == 0.0
    # ... and only becomes 1 the following day.
    assert df["pos"].iloc[3] == 1.0
    # So the 10% return spike on day 2 must NOT show up in ret_gross on day 2.
    assert df["ret_gross"].iloc[2] == 0.0


def test_zero_cost_means_gross_equals_net():
    idx = _idx(5)
    ret = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=idx)
    signal = pd.Series([1, 1, 0, 1, 1], index=idx)
    res = backtest(ret, signal, fees_bps=0.0, slippage_bps=0.0)
    pd.testing.assert_series_equal(res.df["ret_gross"], res.df["ret_net"], check_names=False)


def test_turnover_and_cost_scale_with_position_changes():
    idx = _idx(4)
    ret = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)
    signal = pd.Series([1, 1, 0, 1], index=idx)
    res = backtest(ret, signal, fees_bps=100.0, slippage_bps=0.0)  # 100bps = 1%
    df = res.df

    # pos = signal shifted by 1: [0, 1, 1, 0]
    np.testing.assert_allclose(df["pos"].values, [0.0, 1.0, 1.0, 0.0])
    # turnover = |pos_t - pos_{t-1}|: [0, 1, 0, 1]
    np.testing.assert_allclose(df["turnover"].values, [0.0, 1.0, 0.0, 1.0])
    # cost = turnover * 100bps/1e4 = turnover * 0.01
    np.testing.assert_allclose(df["cost"].values, [0.0, 0.01, 0.0, 0.01])


def test_equity_and_drawdown_on_a_simple_known_path():
    idx = _idx(3)
    ret = pd.Series([0.10, -0.10, 0.0], index=idx)
    signal = pd.Series([1, 1, 1], index=idx)  # always long, ignore the first bar (no prior signal)
    res = backtest(ret, signal, fees_bps=0.0, slippage_bps=0.0)
    df = res.df

    # pos = [0, 1, 1] -> ret_net = [0, -0.10, 0.0]
    expected_equity = (1.0 + df["ret_net"]).cumprod()
    pd.testing.assert_series_equal(df["equity_net"], expected_equity, check_names=False)

    expected_dd = df["equity_net"] / df["equity_net"].cummax() - 1.0
    pd.testing.assert_series_equal(df["drawdown"], expected_dd, check_names=False)
    assert (df["drawdown"] <= 1e-12).all()  # never positive
