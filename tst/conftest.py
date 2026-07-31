"""Shared fixtures for the test suite.

All fixtures are fully synthetic / deterministic (no network access, no
dependency on files under data/), so the suite runs the same way locally
and in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_synthetic_ohlcv(n: int = 400, seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    """Deterministic geometric-random-walk OHLCV series, tz-aware daily index."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    log_rets = rng.normal(loc=0.0003, scale=0.02, size=n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    high = close * (1.0 + np.abs(rng.normal(0, 0.005, size=n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.005, size=n)))
    open_ = close * (1.0 + rng.normal(0, 0.002, size=n))
    volume = rng.integers(1_000, 10_000, size=n).astype(float)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        },
        index=idx,
    )


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    return make_synthetic_ohlcv()
