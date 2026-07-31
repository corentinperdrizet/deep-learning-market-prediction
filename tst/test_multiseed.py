from __future__ import annotations

import numpy as np

from src.validation.multiseed import _mean_ci


def test_mean_ci_basic_stats():
    values = [0.50, 0.52, 0.51, 0.49, 0.53]
    result = _mean_ci(values, confidence=0.95)
    assert result["n"] == 5
    assert result["mean"] == np.mean(values)
    assert result["std"] == np.std(values, ddof=1)
    assert result["ci_low"] < result["mean"] < result["ci_high"]


def test_mean_ci_single_value_has_no_interval():
    result = _mean_ci([0.5])
    assert result["n"] == 1
    assert result["std"] is None
    assert result["ci_low"] is None
    assert result["ci_high"] is None


def test_mean_ci_widens_with_higher_confidence():
    values = [0.40, 0.55, 0.60, 0.45, 0.50, 0.52]
    ci_90 = _mean_ci(values, confidence=0.90)
    ci_99 = _mean_ci(values, confidence=0.99)
    width_90 = ci_90["ci_high"] - ci_90["ci_low"]
    width_99 = ci_99["ci_high"] - ci_99["ci_low"]
    assert width_99 > width_90


def test_mean_ci_shrinks_with_lower_variance():
    tight = [0.50, 0.501, 0.499, 0.5005, 0.4995]
    wide = [0.30, 0.70, 0.40, 0.60, 0.50]
    ci_tight = _mean_ci(tight)
    ci_wide = _mean_ci(wide)
    width_tight = ci_tight["ci_high"] - ci_tight["ci_low"]
    width_wide = ci_wide["ci_high"] - ci_wide["ci_low"]
    assert width_tight < width_wide
