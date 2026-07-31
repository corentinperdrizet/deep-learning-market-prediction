from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import (
    BuyAndHoldClassifier,
    LogisticRegressionTabular,
    SMACrossoverClassifier,
    sequences_to_tabular,
)


def test_sequences_to_tabular_last_and_mean_and_flatten():
    X = np.arange(2 * 4 * 3).reshape(2, 4, 3).astype(float)  # (N=2, T=4, F=3)

    last = sequences_to_tabular(X, pooling="last")
    np.testing.assert_allclose(last, X[:, -1, :])

    mean = sequences_to_tabular(X, pooling="mean")
    np.testing.assert_allclose(mean, X.mean(axis=1))

    flat = sequences_to_tabular(X, pooling="flatten_last_k", k=2)
    assert flat.shape == (2, 2 * 3)
    np.testing.assert_allclose(flat, X[:, -2:, :].reshape(2, -1))


def test_sequences_to_tabular_rejects_bad_k():
    X = np.zeros((2, 3, 4))
    with pytest.raises(ValueError):
        sequences_to_tabular(X, pooling="flatten_last_k", k=0)
    with pytest.raises(ValueError):
        sequences_to_tabular(X, pooling="flatten_last_k", k=10)


def test_buy_and_hold_predicts_train_prevalence():
    y_train = np.array([1, 1, 1, 0])  # prevalence 0.75
    clf = BuyAndHoldClassifier().fit(y_train)
    proba = clf.predict_proba(5)
    assert proba.shape == (5, 2)
    np.testing.assert_allclose(proba[:, 1], 0.75)
    np.testing.assert_allclose(proba[:, 0], 0.25)


def test_buy_and_hold_predict_before_fit_raises():
    clf = BuyAndHoldClassifier()
    with pytest.raises(RuntimeError):
        clf.predict_proba(3)


def test_logistic_regression_tabular_predict_proba_sums_to_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 6, 3))
    y = (X[:, -1, 0] > 0).astype(int)  # separable on the last-timestep first feature
    clf = LogisticRegressionTabular(pooling="last").fit(X, y)
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)
    # Should do meaningfully better than chance on a separable signal.
    acc = ((proba[:, 1] > 0.5).astype(int) == y).mean()
    assert acc > 0.8


def test_sma_crossover_requires_short_less_than_long():
    with pytest.raises(ValueError):
        SMACrossoverClassifier(lookback_short=200, lookback_long=50).fit(
            pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3))
        )


def test_sma_crossover_signals_long_when_short_above_long():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    prices = pd.Series(np.linspace(1, 10, 10), index=idx)  # strictly increasing
    clf = SMACrossoverClassifier(lookback_short=2, lookback_long=4).fit(prices)
    proba = clf.predict_proba(idx)
    # For a strictly increasing series, the short SMA should end up above the
    # long SMA on the later dates -> predicted p(up) = 1.
    assert proba[-1, 1] == 1.0
