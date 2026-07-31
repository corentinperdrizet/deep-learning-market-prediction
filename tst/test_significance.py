from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.significance import (
    block_bootstrap_sharpe_ci,
    cost_sensitivity,
    probabilistic_sharpe_ratio,
)


def test_psr_is_half_when_observed_equals_benchmark():
    """Regardless of n, skew, kurtosis: if the observed Sharpe exactly equals
    the benchmark, there's a coin-flip's worth of evidence it's actually
    better -- PSR must be exactly 0.5.
    """
    # skew/kurtosis combinations kept within the domain where the PSR
    # denominator stays positive for sharpe=1.2 (not every combination is
    # valid -- e.g. large positive skew with kurtosis=3 makes the variance
    # term negative, which is a property of the formula, not a bug).
    for n in [10, 100, 5000]:
        for skew in [-0.5, 0.0, 0.5]:
            psr = probabilistic_sharpe_ratio(1.2, benchmark_sharpe=1.2, n=n, skew=skew, kurtosis=3.0)
            assert psr == pytest.approx(0.5, abs=1e-9)


def test_psr_increases_with_sample_size_when_sharpe_beats_benchmark():
    psr_small_n = probabilistic_sharpe_ratio(1.0, benchmark_sharpe=0.0, n=30)
    psr_large_n = probabilistic_sharpe_ratio(1.0, benchmark_sharpe=0.0, n=3000)
    assert psr_large_n > psr_small_n
    assert psr_small_n > 0.5  # still favors the observed Sharpe, just less confidently


def test_psr_rejects_n_below_two():
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(1.0, benchmark_sharpe=0.0, n=1)


def test_block_bootstrap_ci_contains_observed_sharpe_for_stationary_returns():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=1000, freq="D")
    r = pd.Series(rng.normal(0.0005, 0.01, size=1000), index=idx)

    result = block_bootstrap_sharpe_ci(r, n_boot=300, block_size=15, seed=1)

    assert result["ci_low"] < result["observed_sharpe"] < result["ci_high"]
    assert result["n_boot_valid"] > 0


def test_block_bootstrap_ci_narrows_with_more_data():
    """Holding the return-generating process fixed, a longer sample gives a
    more precise Sharpe estimate -> a narrower bootstrap CI. (Sharpe CI width
    is not simply proportional to return volatility -- scaling sigma up at
    fixed mu changes the true Sharpe itself, which is a different effect.)
    """

    def make_returns(n, seed):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        return pd.Series(rng.normal(0.0008, 0.02, size=n), index=idx)

    short = make_returns(250, seed=3)
    long = make_returns(4000, seed=3)

    ci_short = block_bootstrap_sharpe_ci(short, n_boot=400, block_size=10, seed=2)
    ci_long = block_bootstrap_sharpe_ci(long, n_boot=400, block_size=10, seed=2)

    width_short = ci_short["ci_high"] - ci_short["ci_low"]
    width_long = ci_long["ci_high"] - ci_long["ci_low"]
    assert width_long < width_short


def test_cost_sensitivity_degrades_monotonically_with_fees():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    ret = pd.Series(0.01, index=idx)
    # Always long with occasional flips -> nonzero turnover so fees actually bite.
    signal = pd.Series(np.tile([1, 1, 0, 1], 25), index=idx)

    df = cost_sensitivity(ret, signal, fee_bps_grid=(0.0, 10.0, 50.0, 200.0))
    assert list(df["fees_bps"]) == [0.0, 10.0, 50.0, 200.0]
    # More fees -> lower (or equal) CAGR, monotonically non-increasing.
    cagrs = df["CAGR"].to_numpy()
    assert np.all(np.diff(cagrs) <= 1e-12)
