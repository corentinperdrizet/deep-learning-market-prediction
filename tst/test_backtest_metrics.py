from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import per_period_cost_from_turnover
from src.backtest.metrics import (
    cagr,
    calmar,
    hit_ratio,
    max_drawdown,
    sharpe,
    sortino,
)


def test_cagr_on_constant_daily_return():
    idx = pd.date_range("2020-01-01", periods=252, freq="D")
    r = pd.Series(0.0, index=idx)
    assert cagr(r, periods_per_year=252) == pytest.approx(0.0, abs=1e-9)

    # CAGR = (prod(1+r))^(periods_per_year/n) - 1; with n == periods_per_year
    # this collapses to prod(1+r) - 1, i.e. the full-sample compounded return.
    r2 = pd.Series(0.01, index=idx)
    expected = (1.01**252) - 1.0
    assert cagr(r2, periods_per_year=252) == pytest.approx(expected, rel=1e-9)


def test_sharpe_nan_when_returns_have_zero_volatility():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    r = pd.Series(0.01, index=idx)
    # zero volatility -> undefined Sharpe -> function returns nan
    assert np.isnan(sharpe(r, periods_per_year=252))


def test_sharpe_positive_for_positive_mean_positive_vol():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500, freq="D")
    r = pd.Series(rng.normal(0.001, 0.01, size=500), index=idx)
    s = sharpe(r, periods_per_year=252)
    assert s > 0


def test_sortino_ignores_upside_volatility():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    # All positive returns of varying magnitude -> downside deviation is 0 -> nan
    r = pd.Series([0.01, 0.05, 0.02, 0.09, 0.01, 0.03], index=idx)
    assert np.isnan(sortino(r, periods_per_year=252))


def test_max_drawdown_on_known_equity_path():
    equity = pd.Series([1.0, 1.2, 0.9, 1.1, 0.6, 1.5])
    mdd = max_drawdown(equity)
    # worst drawdown: from peak 1.2 down to 0.6 -> 0.6/1.2 - 1 = -0.5
    assert mdd == pytest.approx(-0.5, rel=1e-9)


def test_calmar_ratio_relationship_to_cagr_and_mdd():
    idx = pd.date_range("2020-01-01", periods=252, freq="D")
    # A mix of gains and one sharp drop, so max_drawdown is a real, nonzero number.
    r = pd.Series(0.001, index=idx)
    r.iloc[100] = -0.20
    equity = (1 + r).cumprod()
    c = cagr(r, periods_per_year=252)
    mdd = abs(max_drawdown(equity))
    assert mdd > 0
    expected = c / mdd
    assert calmar(r, equity=equity, periods_per_year=252) == pytest.approx(expected, rel=1e-9)


def test_calmar_is_nan_when_there_is_no_drawdown():
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    r = pd.Series(0.001, index=idx)  # strictly increasing equity -> zero drawdown
    equity = (1 + r).cumprod()
    assert np.isnan(calmar(r, equity=equity, periods_per_year=252))


def test_hit_ratio_counts_positive_periods():
    r = pd.Series([0.01, -0.01, 0.02, 0.0, -0.005])
    assert hit_ratio(r) == pytest.approx(2 / 5)


def test_per_period_cost_from_turnover_linear_in_bps():
    turnover = pd.Series([0.0, 1.0, 0.5])
    cost = per_period_cost_from_turnover(turnover, fees_bps=10.0, slippage_bps=5.0)
    # (10+5)bps = 15bps = 0.0015
    np.testing.assert_allclose(cost.values, [0.0, 0.0015, 0.00075])
