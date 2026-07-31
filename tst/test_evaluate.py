"""Regression test for a real bug found in review: src/training/evaluate.py
used `sys.stderr` in its exception-handling paths without importing `sys`,
so run_baselines() crashed with a NameError instead of gracefully skipping
the SMA baseline whenever the price series was missing or misaligned.
"""

from __future__ import annotations

import numpy as np

from src.training.evaluate import evaluate_classifier, run_baselines


def _tiny_dataset():
    rng = np.random.default_rng(0)
    n_train, n_val, n_test, seq_len, n_feat = 60, 20, 20, 5, 3
    return {
        "X_train": rng.normal(size=(n_train, seq_len, n_feat)),
        "y_train": rng.integers(0, 2, size=n_train),
        "X_val": rng.normal(size=(n_val, seq_len, n_feat)),
        "y_val": rng.integers(0, 2, size=n_val),
        "X_test": rng.normal(size=(n_test, seq_len, n_feat)),
        "y_test": rng.integers(0, 2, size=n_test),
        "idx": {},  # no datetime index provided -> forces the "skip SMA" path
    }


def test_run_baselines_without_prices_does_not_crash():
    """prices=None makes _coerce_prices() return None, which used to hit the
    buggy `print(..., file=sys.stderr)` branch and raise NameError.
    """
    dataset = _tiny_dataset()
    df = run_baselines(dataset, prices=None, use_xgb=False)
    assert set(df["model"].unique()) >= {"buy_hold", "logreg[last]"}
    assert "sma_50_200" not in set(df["model"].unique())


def test_run_baselines_returns_val_and_test_rows_for_every_model():
    dataset = _tiny_dataset()
    df = run_baselines(dataset, prices=None)
    counts = df.groupby("model")["split"].nunique()
    assert (counts == 2).all()


def test_evaluate_classifier_metrics_keys_and_ranges():
    y_true = np.array([0, 1, 0, 1, 1])
    y_proba = np.stack([1 - np.array([0.1, 0.8, 0.3, 0.6, 0.9]), np.array([0.1, 0.8, 0.3, 0.6, 0.9])], axis=1)
    m = evaluate_classifier(y_true, y_proba)
    for key in ["accuracy", "f1_pos", "roc_auc", "pr_auc", "brier"]:
        assert key in m
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["brier"] <= 1.0
