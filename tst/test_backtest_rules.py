from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.rules import signal_from_proba, signal_from_regression


def test_signal_from_proba_long_flat():
    p = pd.Series([0.9, 0.5, 0.1, 0.6])
    sig = signal_from_proba(p, theta=0.55, long_short=False)
    np.testing.assert_allclose(sig.values, [1.0, 0.0, 0.0, 1.0])


def test_signal_from_proba_long_short():
    p = pd.Series([0.9, 0.5, 0.1])
    sig = signal_from_proba(p, theta=0.6, long_short=True)
    # p=0.9 > theta -> +1 ; p=0.1 < 1-theta=0.4 -> -1 ; p=0.5 -> flat
    np.testing.assert_allclose(sig.values, [1.0, 0.0, -1.0])


@pytest.mark.parametrize("theta", [0.0, 1.0, -0.1, 1.5])
def test_signal_from_proba_rejects_invalid_theta(theta):
    with pytest.raises(ValueError):
        signal_from_proba(pd.Series([0.5]), theta=theta)


def test_signal_from_regression_clips_and_scales():
    y_pred = pd.Series([0.5, -2.0, 0.1, 3.0])
    pos = signal_from_regression(y_pred, k=1.0, clip=1.0)
    np.testing.assert_allclose(pos.values, [0.5, -1.0, 0.1, 1.0])

    pos_scaled = signal_from_regression(y_pred, k=2.0, clip=1.0)
    np.testing.assert_allclose(pos_scaled.values, [1.0, -1.0, 0.2, 1.0])
