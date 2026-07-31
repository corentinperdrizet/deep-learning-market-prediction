from __future__ import annotations

import numpy as np

from src.training.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    calibration_report,
    expected_calibration_error,
)


def _synthetic_miscalibrated(n=2000, seed=0):
    """Ground-truth probabilities q, but the model reports an over-confident,
    monotonically related p = sigmoid(3 * logit(q)) -- a classic case Platt
    scaling / isotonic regression should be able to correct.
    """
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.05, 0.95, size=n)
    y = (rng.uniform(size=n) < q).astype(float)
    logit_q = np.log(q / (1 - q))
    p_overconfident = 1.0 / (1.0 + np.exp(-3.0 * logit_q))
    return y, p_overconfident


def test_platt_calibration_reduces_ece():
    y, p = _synthetic_miscalibrated()
    ece_before, _, _ = expected_calibration_error(y, p)

    cal = PlattCalibrator().fit(y, p)
    p_cal = cal.transform(p)
    ece_after, _, _ = expected_calibration_error(y, p_cal)

    assert ece_after < ece_before


def test_isotonic_calibration_is_monotonic_in_input_order():
    y, p = _synthetic_miscalibrated()
    cal = IsotonicCalibrator().fit(y, p)
    order = np.argsort(p)
    p_cal_sorted = cal.transform(p[order])
    assert np.all(np.diff(p_cal_sorted) >= -1e-12)


def test_expected_calibration_error_near_zero_for_perfectly_calibrated_probs():
    rng = np.random.default_rng(1)
    n = 20000
    p = rng.uniform(0.0, 1.0, size=n)
    y = (rng.uniform(size=n) < p).astype(float)
    ece, mce, table = expected_calibration_error(y, p, n_bins=10)
    assert ece < 0.03
    assert set(table.columns) == {"bin", "count", "accuracy", "confidence", "gap"}


def test_calibration_report_keys_and_ranges():
    y, p = _synthetic_miscalibrated()
    report = calibration_report(y, p)
    assert set(report.keys()) == {"ECE", "MCE", "Brier"}
    assert 0.0 <= report["ECE"] <= 1.0
    assert 0.0 <= report["Brier"] <= 1.0
